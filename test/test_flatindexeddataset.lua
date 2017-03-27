--
-- Copyright 2017-present Facebook.
-- All Rights Reserved.
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.
--

local tnt = require 'torchnet.sequential'

local tester
local test = torch.TestSuite()

local data = {
    torch.IntTensor({1, 2, 3, 4, 5}),
    torch.IntTensor({6, 7}),
    torch.IntTensor({8, 9, 10, 11}),
    torch.IntTensor({12}),
    torch.IntTensor({13, 14, 15, 16, 17, 18}),
    torch.IntTensor({19}),
    torch.IntTensor({20}),
    torch.IntTensor({21, 22}),
    torch.IntTensor({23, 24, 25, 26}),
}

function test.FlatIndexed_Equal()
    -- XXX A temporary directory function would be great
    local dest = os.tmpname()
    local writer = tnt.IndexedDatasetWriter{
        indexfilename = dest .. '.idx',
        datafilename = dest .. '.bin',
        type = 'int',
    }

    for _, t in ipairs(data) do
        writer:add(t)
    end
    writer:close()

    local field = paths.basename(dest)
    local ds = tnt.IndexedDataset{
        fields = {field},
        path = paths.dirname(dest),
    }
    tester:assertGeneralEq(#data, ds:size())

    local fds = tnt.FlatIndexedDataset{
        indexfilename = dest .. '.idx',
        datafilename = dest .. '.bin',
        indices = true,
    }
    local totals = 0
    for i = 1, ds:size() do
        totals = totals + ds:get(i)[field]:nElement()
    end
    tester:assertGeneralEq(totals, fds:size())

    -- Check elements and indices to original data
    local j = 1
    for i = 1, ds:size() do
        local s = ds:get(i)[field]:view(-1)
        for k = 1, s:nElement() do
            local v, idx = fds:get(j)
            tester:assertGeneralEq(s[k], v)
            tester:assertGeneralEq(i, idx)
            j = j + 1
        end
    end
end

function test.FlatIndexed_LowerBound()
    local t = torch.IntTensor({1, 2, 5, 7, 8})
    local lbound = tnt.FlatIndexedDataset._lowerBound

    tester:assertGeneralEq(1, lbound(t, 1))
    tester:assertGeneralEq(2, lbound(t, 2))
    tester:assertGeneralEq(2, lbound(t, 3))
    tester:assertGeneralEq(2, lbound(t, 4))
    tester:assertGeneralEq(3, lbound(t, 5))
    tester:assertGeneralEq(3, lbound(t, 6))
    tester:assertGeneralEq(4, lbound(t, 7))
    tester:assertGeneralEq(5, lbound(t, 8))
    -- Out-of-bounds queries are not handled by this function.

    tester:assertGeneralEq(0, lbound(torch.IntTensor(), 1))
end

return function(_tester_)
    tester = _tester_
    return test
end
