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

local ds = tnt.TableDataset{data = {
    torch.rand(5),
    torch.rand(3),
    torch.rand(1),
    torch.rand(7),
}}

function test.TruncatedIterator_Simple()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
        maxsize = 3,
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    for s in it() do
        its = its + 1
        tester:assertGeneralEq(partlens[its], s:size(1))
    end
    tester:assertGeneralEq(#partlens, its)
end

function test.TruncatedIterator_SimpleFn()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
        maxsizefn = function(s) return 3 end,
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    for s in it() do
        its = its + 1
        tester:assertGeneralEq(partlens[its], s:size(1))
    end
    tester:assertGeneralEq(#partlens, its)
end

function test.TruncatedIterator_Iterator()
    local it = tnt.TruncatedDatasetIterator{
        iterator = tnt.DatasetIterator{
            dataset = ds,
        },
        maxsize = 3,
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    for s in it() do
        its = its + 1
        tester:assertGeneralEq(partlens[its], s:size(1))
    end
    tester:assertGeneralEq(#partlens, its)
end

function test.TruncatedIterator_IteratorFn()
    local it = tnt.TruncatedDatasetIterator{
        iterator = tnt.DatasetIterator{
            dataset = ds,
        },
        maxsizefn = function(s)
            if s:size(1) == 7 then
                return math.huge  -- don't split
            end
            return 3
        end,
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 7}
    for s in it() do
        its = its + 1
        tester:assertGeneralEq(partlens[its], s:size(1))
    end
    tester:assertGeneralEq(#partlens, its)
end

function test.TruncatedIterator_Single()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
        maxsize = 1,
    }

    local its = 0
    for s in it() do
        its = its + 1
        tester:assertGeneralEq(1, s:size(1))
    end
    tester:assertGeneralEq(16, its)
end

function test.TruncatedIterator_TableContSplit()
    local it = tnt.TruncatedDatasetIterator{
        dataset = tnt.TransformDataset{
            dataset = ds,
            transform = function(sample) return {x = sample} end
        },
        maxsize = 3,
        fields = {'x'},
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    local conts = {false, true, false, false, false, true, true}
    local nexts = {true, false, false, false, true, true, false}
    local splits = {true, true, false, false, true, true, true}
    for s in it() do
        its = its + 1
        tester:assertGeneralEq(partlens[its], s.x:size(1))
        tester:assertGeneralEq(conts[its], s._cont == true)
        tester:assertGeneralEq(nexts[its], s._hasnext == true)
        tester:assertGeneralEq(splits[its], s._split == true)
    end
    tester:assertGeneralEq(#partlens, its)
end

function test.TruncatedIterator_TableNoFields()
    local it = tnt.TruncatedDatasetIterator{
        dataset = tnt.TransformDataset{
            dataset = ds,
            transform = function(sample) return {x = sample} end
        },
        maxsize = 3,
    }

    local its = 0
    for s in it() do
        its = its + 1
        tester:assertGeneralEq(ds:get(its):totable(), s.x:totable())
    end
    tester:assertGeneralEq(ds:size(), its)
end

function test.TruncatedIterator_Exclude()
    local it = tnt.TruncatedDatasetIterator{
        dataset = tnt.TransformDataset{
            dataset = ds,
            transform = function(sample) return {x = sample, y = sample} end
        },
        maxsize = 3,
        fields = {'x'}
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    local excllens = {5, 5, 3, 1, 7, 7, 7}
    local cons = {false, true, false, false, false, true, true}
    local nexts = {true, false, false, false, true, true, false}
    for s in it() do
        its = its + 1
        tester:assertGeneralEq(partlens[its], s.x:size(1))
        tester:assertGeneralEq(excllens[its], s.y:size(1))
        tester:assertGeneralEq(cons[its], s._cont == true)
        tester:assertGeneralEq(nexts[its], s._hasnext == true)
    end
    tester:assertGeneralEq(#partlens, its)
end

function test.TruncatedIterator_MixedLength()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
        dataset = tnt.TransformDataset{
            dataset = ds,
            transform = function(sample)
                return {x = sample, y = torch.cat(sample, sample)}
            end
        },
        maxsize = 3,
        fields = {'x', 'y'}
    }

    -- This will trigger an assertion since x and y are of different
    -- length
    tester:assertError(it())
end

function test.TruncatedIterator_Batch()
    local it = tnt.TruncatedDatasetIterator{
        dataset = tnt.TableDataset{data = {
            torch.rand(1, 5),
            torch.rand(2, 3),
            torch.rand(3, 1),
            torch.rand(4, 7),
        }},
        maxsize = 3,
        dimension = 2,
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    local batchlens = {1, 1, 2, 3, 4, 4, 4}
    for s in it() do
        its = its + 1
        tester:assertGeneralEq(partlens[its], s:size(2))
        tester:assertGeneralEq(batchlens[its], s:size(1))
    end
    tester:assertGeneralEq(#partlens, its)
end

function test.TruncatedIterator_MinSize()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
        maxsize = 3,
        minsize = 2,
    }

    local its = 0
    local partlens = {3, 2, 3, 3, 3}
    for s in it() do
        its = its + 1
        tester:assertGeneralEq(partlens[its], s:size(1))
    end
    tester:assertGeneralEq(#partlens, its)
end

function test.TruncatedIterator_Identity()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
    }

    local its = 0
    for s in it() do
        its = its + 1
        tester:assertGeneralEq(ds:get(its):totable(), s:totable())
    end
    tester:assertGeneralEq(ds:size(), its)
end

return function(_tester_)
    tester = _tester_
    return test
end
