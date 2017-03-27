# torchnet/sequential
This package contains
[torchnet](https://github.com/torchnet/torchnet)-compatible utilities for
working with sequential data. In time-series processing such as for natural
language or speech, it is often necessary to add a good amount of boilerplate
code, e.g. for pre-processing the data into mini-batches. torchnet/sequential
aims at providing small and modular building blocks to alleviate the need for
boilerplate code while retaining a flexible setup for easy experimentation.

# Installation
Install [torchnet](https://github.com/torchnet/torchnet) first, and then do
```
luarocks install rocks/torchnet-sequential-*.rockspec
```

# Package contents
## Datasets
torchnet/sequential provides multiple re-usable torchnet dataset classes for
composing flexible pre-processing pipelines, e.g. for two-dimensional
mini-batching (across time and across multiple sequences) or truncated BPTT
(back-propagation through time) training.

What follows is a brief description of the dataset classes. Please refer to the
[dataset README](https://github.com/torchnet/sequential/tree/master/dataset)
and the source code for further details, discussion and example use cases.

### tnt.FlatIndexedDataset
This dataset provides a "flat" view of the storage that would usually back a
`tnt.IndexedDataset`. In other words, this dataset disregards the index on disk
and provides access to the individual data elements.

### tnt.SequenceBatchDataset
This dataset produces multiple, parallel sequences from a single sequence
represented by the underlying dataset.

### tnt.BucketSortedDataset
This is a resampling dataset that clusters samples into buckets based on their
size. By customizing the `samplesize` function and setting the `resolution`
argument, various bucketing and sharding patterns can be implemented.

### tnt.BucketBatchDataset
This class resamples a dataset using BucketSortedDataset and yields batches from
within buckets. Similar to `tnt.BatchDataset` a policy is used to control the
behavior regarding remaining samples in a bucket that do not fill up a whole
batch.

### tnt.TargetNextDataset
This dataset produces augments samples with successive samples, assuming that
the underlying dataset represents a sequence. Samples at the end of the
underlying dataset will be discarded if they don't have a corresponding target.
This is useful for generating targets for language modeling tasks.

### tnt.TruncatedDatasetIterator
This iterator truncates samples produced by its underlying dataset or iterator
if they exceed the specified maximum size. In this case, multiple smaller
samples are produced. This iterator can be used to enforce a limit on
backpropagation through time.

## License
torchnet/sequential is BSD-licensed. We also provide an additional patent grant.
