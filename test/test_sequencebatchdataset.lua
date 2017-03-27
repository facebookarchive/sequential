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

function test.SequenceBatch_Simple()
    local src = tnt.TableDataset{data = torch.linspace(1, 14, 14):totable()}
    local bsz = 4
    local ds = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = bsz,
    }

    tester:assertGeneralEq(math.ceil(src:size() / bsz), ds:size())
    tester:assertGeneralEq(bsz, ds:get(1):size(1))
    tester:assertGeneralEq(bsz, ds:get(1):nElement())
    tester:assertGeneralEq({1, 5, 9, 13}, ds:get(1):totable())
    tester:assertGeneralEq({2, 6, 10, 14}, ds:get(2):totable())
    tester:assertGeneralEq({3, 7, 11, 1}, ds:get(3):totable())
    tester:assertGeneralEq({4, 8, 12, 1}, ds:get(4):totable())
    tester:assertError(function() ds:get(5) end)
    tester:assertError(function() ds:get(0) end)
end

function test.SequenceBatch_Types()
    local src = tnt.TableDataset{data = torch.linspace(1, 14, 14):totable()}
    local bsz = 4

    -- Try a few common types
    local types = {'torch.IntTensor', 'torch.LongTensor', 'torch.FloatTensor',
        'torch.DoubleTensor'}
    for _,t in ipairs(types) do
        tester:assertGeneralEq(t, tnt.SequenceBatchDataset{
                dataset = src,
                batchsize = bsz,
                type = t,
            }:get(1):type()
        )
    end

    local dsF = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = bsz,
        type = 'nosuchtype',
    }
    -- Access should fail when trying to create a tensor with the nonexistent
    -- type
    tester:assertError(function() dsF:get(1) end)
end

function test.SequenceBatch_Extremes()
    local src = tnt.TableDataset{data = torch.linspace(1, 14, 14):totable()}

    tester:assertError(function() tnt.SequenceBatchDataset(src, 0) end)

    local dsN = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = src:size(),
    }
    tester:assertGeneralEq(1, dsN:size())
    tester:assertGeneralEq(src:size(), dsN:get(1):nElement())

    local ds2N = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = src:size() * 2,
    }
    tester:assertGeneralEq(1, ds2N:size())
    tester:assertGeneralEq(src:size() * 2, ds2N:get(1):nElement())

    local ds2NS = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = src:size() * 2,
        policy = 'skip-remainder',
    }
    tester:assertGeneralEq(0, ds2NS:size())
end

function test.SequenceBatch_Policy()
    local src = tnt.TableDataset{data = torch.linspace(1, 14, 14):totable()}

    local dsSL = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = src:size() - 1,
        policy = 'skip-remainder',
    }
    tester:assertGeneralEq(1, dsSL:size())
    tester:assertGeneralEq(torch.linspace(1, 13, 13):totable(),
        dsSL:get(1):totable())

    local dsIL = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = src:size() - 1,
        policy = 'pad-remainder',
        pad = 0,
    }
    tester:assertGeneralEq(2, dsIL:size())
    tester:assertGeneralEq(
        torch.cat(torch.range(1, 13, 2), torch.zeros(6)):totable(),
        dsIL:get(1):totable()
    )
    tester:assertGeneralEq(
        torch.cat(torch.range(2, 14, 2), torch.zeros(6)):totable(),
        dsIL:get(2):totable()
    )
end

return function(_tester_)
    tester = _tester_
    return test
end
