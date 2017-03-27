--
-- Copyright 2017-present Facebook.
-- All Rights Reserved.
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.
--

local __main__ = package.loaded['torchnet.env'] == nil
local _ = require 'torchnet.env'

if __main__ then
    require 'torchnet'
end

local tester = torch.Tester()
tester:add(paths.dofile('test_bucketsorteddataset.lua')(tester))
tester:add(paths.dofile('test_bucketbatchdataset.lua')(tester))
tester:add(paths.dofile('test_flatindexeddataset.lua')(tester))
tester:add(paths.dofile('test_targetnextdataset.lua')(tester))
tester:add(paths.dofile('test_sequencebatchdataset.lua')(tester))
tester:add(paths.dofile('test_truncateddatasetiterator.lua')(tester))

local function dotest(tests)
    tester:run(tests)
    return tester
end

if __main__ then
    require 'torchnet'
    if #arg > 0 then
        dotest(arg)
    else
        dotest()
    end
else
    return tester
end
