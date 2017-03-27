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

function test.BucketBatch_Simple()
    local data = {
        "a", "b", "c",
        "aa", "bb", "cc",
        "aaaa", "bbbb", "cccc",
        "foobar",
    }

    torch.manualSeed(1234)
    local ds = tnt.BucketBatchDataset{
        dataset = tnt.TransformDataset{
            dataset = tnt.TableDataset{data = data},
            transform = function(sample) return {s=sample} end,
        },
        batchsize = 2,
        merge = function(tbl) return {s=table.concat(tbl.s, ',')} end,
        samplesize = function(ds, i) return #ds:get(i).s end,
        shuffle = true
    }
    tester:assertGeneralEq(7, ds:size())
    tester:assertGeneralEq("a,c", ds:get(1).s)
    tester:assertGeneralEq("b", ds:get(2).s)
    tester:assertGeneralEq("aa,cc", ds:get(3).s)
    tester:assertGeneralEq("bb", ds:get(4).s)
    tester:assertGeneralEq("cccc,bbbb", ds:get(5).s)
    tester:assertGeneralEq("aaaa", ds:get(6).s)
    tester:assertGeneralEq("foobar", ds:get(7).s)

    ds:resampleInBuckets()
    tester:assertGeneralEq("a,c", ds:get(1).s)
    tester:assertGeneralEq(nil, ds:get(2).s:find(','))
    tester:assertGeneralEq(nil, ds:get(4).s:find(','))
    tester:assertGeneralEq(nil, ds:get(6).s:find(','))
    tester:assertGeneralEq(nil, ds:get(7).s:find(','))

    local ds2 = tnt.BucketBatchDataset{
        dataset = tnt.TransformDataset{
            dataset = tnt.TableDataset{data = data},
            transform = function(sample) return {s=sample} end,
        },
        batchsize = 4,
        merge = function(tbl) return {s=table.concat(tbl.s, ',')} end,
        resolution = 3,
        samplesize = function(ds, i) return #ds:get(i).s end,
    }
    tester:assertGeneralEq(4, ds2:size())
    tester:assertGeneralEq("a,b,c,aa", ds2:get(1).s)
    tester:assertGeneralEq("bb,cc", ds2:get(2).s)
    tester:assertGeneralEq("aaaa,bbbb,cccc", ds2:get(3).s)
    tester:assertGeneralEq("foobar", ds2:get(4).s)
end

function test.BucketBatch_IncludeLast()
    local data = torch.linspace(0, 99, 100):long()
    local ds = tnt.BucketBatchDataset{
        dataset = tnt.TransformDataset{
            dataset = tnt.TableDataset{data = data:totable()},
            transform = function(sample) return {s=sample} end,
        },
        batchsize = 20,
        resolution = 50,
        merge = function(tbl) return {s=torch.LongTensor(tbl.s)} end,
        samplesize = function(ds, i) return ds:get(i).s end,
        policy = 'include-last',
    }

    tester:assertGeneralEq(6, ds:size())
    tester:assertGeneralEq(20, ds:get(1).s:size(1))
    tester:assertGeneralEq(20, ds:get(2).s:size(1))
    tester:assertGeneralEq(10, ds:get(3).s:size(1))
    tester:assertGeneralEq(20, ds:get(4).s:size(1))
    tester:assertGeneralEq(20, ds:get(5).s:size(1))
    tester:assertGeneralEq(10, ds:get(6).s:size(1))
end

function test.BucketBatch_SkipLast()
    local data = torch.linspace(0, 99, 100):long()
    local ds = tnt.BucketBatchDataset{
        dataset = tnt.TransformDataset{
            dataset = tnt.TableDataset{data = data:totable()},
            transform = function(sample) return {s=sample} end,
        },
        batchsize = 20,
        resolution = 50,
        merge = function(tbl) return {s=torch.LongTensor(tbl.s)} end,
        samplesize = function(ds, i) return ds:get(i).s end,
        policy = 'skip-last',
    }

    tester:assertGeneralEq(4, ds:size())
    tester:assertGeneralEq(20, ds:get(1).s:size(1))
    tester:assertGeneralEq(20, ds:get(2).s:size(1))
    tester:assertGeneralEq(20, ds:get(3).s:size(1))
    tester:assertGeneralEq(20, ds:get(4).s:size(1))
end

local function doTestEven(policy)
    local data = torch.linspace(0, 11, 12):long()
    local ds = tnt.BucketBatchDataset{
        dataset = tnt.TransformDataset{
            dataset = tnt.TableDataset{data = data:totable()},
            transform = function(sample) return {s=sample} end,
        },
        batchsize = 2,
        resolution = 4,
        merge = function(tbl) return {s=torch.LongTensor(tbl.s)} end,
        samplesize = function(ds, i) return ds:get(i).s end,
        policy = policy,
    }

    tester:assertGeneralEq(6, ds:size())
    tester:assertGeneralEq(2, ds:get(1).s:size(1))
    tester:assertGeneralEq(2, ds:get(2).s:size(1))
    tester:assertGeneralEq(2, ds:get(3).s:size(1))
    tester:assertGeneralEq(2, ds:get(4).s:size(1))
    tester:assertGeneralEq(2, ds:get(5).s:size(1))
    tester:assertGeneralEq(2, ds:get(6).s:size(1))
end

function test.BucketBatch_EvenIncludeLast()
    doTestEven('include-last')
end

function test.BucketBatch_EvenSkipLast()
    doTestEven('skip-last')
end

return function(_tester_)
    tester = _tester_
    return test
end
