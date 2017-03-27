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

function test.TargetNext_Simple()
    local src = tnt.TransformDataset{
        dataset = tnt.TableDataset{data = torch.linspace(1, 10, 10):totable()},
        transform = function(sample) return {input = sample} end
    }
    local step = 2
    local ds = tnt.TargetNextDataset{
        dataset = src,
        step = step,
    }

    tester:assertGeneralEq(src:size() - step, ds:size())
    tester:assertGeneralEq({input = 1, target = 3}, ds:get(1))
    tester:assertGeneralEq({input = 2, target = 4}, ds:get(2))
    tester:assertGeneralEq({input = 3, target = 5}, ds:get(3))
    tester:assertError(function() return ds:get(0) end)
    tester:assertError(function() return ds:get(src:size()) end)
end

function test.TargetNext_Extremes()
    local src = tnt.TransformDataset{
        dataset = tnt.TableDataset{data = torch.linspace(1, 10, 10):totable()},
        transform = function(sample) return {input = sample} end
    }

    local ds0 = tnt.TargetNextDataset{
        dataset = src,
        step = 0
    }
    tester:assertGeneralEq(src:size(), ds0:size())
    tester:assertGeneralEq({input = 1, target = 1}, ds0:get(1))

    local dsN = tnt.TargetNextDataset{
        dataset = src,
        step = src:size()-1
    }
    tester:assertGeneralEq(1, dsN:size())
    tester:assertGeneralEq({input = 1, target = 10}, dsN:get(1))

    local ds2N = tnt.TargetNextDataset{
        dataset = src,
        step = src:size() * 2
    }
    tester:assertGeneralEq(0, ds2N:size())
end

return function(_tester_)
    tester = _tester_
    return test
end
