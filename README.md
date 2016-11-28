# Notebook Feature Extractor
Notebook Feature Extractor takes an arbitrary collection of IPython
notebooks and generates one vector of floats per notebook,
suitable for wrapping in a Pandas Dataframe.

Values in the vector are currently the average value per-cell of mostly
boolean analysis functions, cast to 1.0 or 0.0. There's also a
lines-per-cell function.

Problems:
If a file doesn't have code cells it doesn't extract features
- rare but important case

Primary TODO:
- figure out which features meaningfully identify notebook characteristics

We should add some notebook-wise features: are cells in order?

Alternate approaches to explore:
- within a notebook, create a vector from each cell
  - analyze at notebook level or have notebooks be multi-dimensional vectors
  - pivot the analysis, building the vectors cellwise rather than featurewise

More features to add:
- cellwise:
  - writes data to:?
- repo-wise:
  - cluster repo:
    - examine lengths of intersections of the sets of cell hashes
    - relative to the overall notebook lengths
    - maybe also check basenames

Other TODO:
- work with other notebook versions
- write unit tests for features
  - base on cells extracted from working notebooks
- cellwise: actually compile code and inspect instead of regex/string match

## Invocation
The code can be invoced in the following manner...

```bash
$ python feature-extractor.py <input directory> <output csv file>
```

## Example Notebook
This repository also contains a sample notebook that performs clustering on various public notebooks from @mozilla staff. **NOTE:** notebook requires original csv file. Do not run unless you have the csv file.
