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

function test.BucketSorted_Simple()
    local data = {
        "a", "b", "c",
        "foobar",
        "aa", "bb",
        "aaaa", "bbbb", "cccc",
        "cc",
    }

    local ds = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data},
    }
    tester:assertGeneralEq(10, ds:size())
    tester:assertGeneralEq("a", ds:get(1))
    tester:assertGeneralEq("b", ds:get(2))
    tester:assertGeneralEq("c", ds:get(3))
    tester:assertGeneralEq("aa", ds:get(4))
    tester:assertGeneralEq("bb", ds:get(5))
    tester:assertGeneralEq("cc", ds:get(6))
    tester:assertGeneralEq("aaaa", ds:get(7))
    tester:assertGeneralEq("bbbb", ds:get(8))
    tester:assertGeneralEq("cccc", ds:get(9))
    tester:assertGeneralEq("foobar", ds:get(10))

    torch.manualSeed(1234)
    local ds1 = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data},
        shuffle = true,
    }
    tester:assertGeneralEq(10, ds1:size())
    tester:assertGeneralEq("a", ds1:get(1))
    tester:assertGeneralEq("c", ds1:get(2))
    tester:assertGeneralEq("b", ds1:get(3))
    tester:assertGeneralEq("aa", ds1:get(4))
    tester:assertGeneralEq("cc", ds1:get(5))
    tester:assertGeneralEq("bb", ds1:get(6))
    tester:assertGeneralEq("cccc", ds1:get(7))
    tester:assertGeneralEq("bbbb", ds1:get(8))
    tester:assertGeneralEq("aaaa", ds1:get(9))
    tester:assertGeneralEq("foobar", ds1:get(10))

    ds1:resampleInBuckets()
    tester:assertGeneralEq("a", ds1:get(1))
    tester:assertGeneralEq("foobar", ds1:get(10))

    local ds2 = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data},
        resolution = 3,
        shuffle = true,
    }
    tester:assertGeneralEq(10, ds2:size())
    tester:assertGeneralEq("bb", ds2:get(1))
    tester:assertGeneralEq("aa", ds2:get(2))
    tester:assertGeneralEq("cc", ds2:get(3))
    tester:assertGeneralEq("b", ds2:get(4))
    tester:assertGeneralEq("a", ds2:get(5))
    tester:assertGeneralEq("c", ds2:get(6))
    tester:assertGeneralEq("aaaa", ds2:get(7))
    tester:assertGeneralEq("bbbb", ds2:get(8))
    tester:assertGeneralEq("cccc", ds2:get(9))
    tester:assertGeneralEq("foobar", ds2:get(10))
end

local function getAllSamples(ds)
    local samples = {}
    for sample in tnt.DatasetIterator(ds)() do
        table.insert(samples, sample)
    end
    return torch.LongTensor(samples)
end

function test.BucketSorted_SingleBucket()
    local data = torch.linspace(1, 100, 100):long()
    local ds = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data:totable()},
        samplesize = function(ds, i) return 1 end,
        shuffle = true,
    }

    torch.manualSeed(3)
    ds:resampleInBuckets()
    tester:assertGeneralEq(2, data:eq(getAllSamples(ds)):sum())

    torch.manualSeed(4)
    ds:resampleInBuckets()
    tester:assertGeneralEq(0, data:eq(getAllSamples(ds)):sum())
end

function test.BucketSorted_OneElemBuckets()
    local data = torch.linspace(1, 100, 100):long()
    local ds = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data:totable()},
        samplesize = function(ds,i) return ds:get(i) end,
        shuffle = true,
    }

    tester:assertGeneralEq(100, data:eq(getAllSamples(ds)):sum())
    ds:resampleInBuckets()
    tester:assertGeneralEq(100, data:eq(getAllSamples(ds)):sum())
end

function test.BucketSorted_NegativeLength()
    local data = torch.linspace(1, 100, 100):long()
    local ds = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data:totable()},
        samplesize = function(ds, i) return -ds:get(i) end,
        shuffle = true,
    }

    local datar = torch.sort(data, 1, true)
    tester:assertGeneralEq(100, datar:eq(getAllSamples(ds)):sum())
end

return function(_tester_)
    tester = _tester_
    return test
end
