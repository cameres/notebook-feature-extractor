import os, re, ujson as json
from collections import defaultdict
import csv


def is_code(cell):
    '''is a code cell'''
    return bool(cell['cell_type'] == 'code')

# TODO: complete
# def is_markdown(cell):
#     '''is a markdown cell'''
#     return bool(cell['cell_type'] == 'markdown')

def has_code(cell):
    '''some cells are code cells that have empty sources
    this is an indicator of a work in progress notebook
        - required to run other feature extractors
    '''
    return bool(cell['source'])

SPARK_NAMES = [
    'pyspark',
    'spark',
    'sqlContext',
    'sqlCtx',
    'SQLContext',
    'SparkContext',
    'SparkSession'
]

# regular expression for matching spark contexts
SPARK_PATTERN = '|'.join(SPARK_NAMES)

def probably_uses_pyspark(cell):
    '''is a cell likely to be using pyspark'''
    # note there is a regular expression issue
    # with variables sc = sklearn.preprocessing.StandardScaler
    if re.search(SPARK_PATTERN, ''.join(cell['source'])):
        return 1
    return 0


def probably_uses_s3(cell): # XXX this is terrible
    '''see if s3 uri is in a cell's source'''
    if 's3://' in ''.join(cell['source']):
        return 1
    return 0


def draws_graph(cell):
    '''see if cell does any type of plotting
        - what about usage with ipython notebook
        magics for matplotlib etc?.
    '''

    if 'plot(' in ''.join(cell['source']):
        return 1
    return 0


def has_traceback(cell):
    '''is a traceback/error found in any of the output?'''
    return any(['traceback' in output for output in cell['outputs']])


def lines(cell):
    '''get the number of lines in a cell'''
    return len([l for l in cell['source'] if l.strip()])

def cell_magic(cell):
    '''magic cell i.e. %%javascript'''
    # cell magic can only be found in first line
    if not has_code(cell):
        return 0
    first_line = cell['source'][0]
    return bool(re.search('^%%', first_line))

def line_magic(cell):
    '''cells with line magics i.e. %timeit ...'''
    if not has_code(cell):
        return 0
    for line in cell['source']:
        # use negative lookahead to only
        # match one %
        if re.search('^%(?!%)', line):
            return 1
    return 0

def line_bang(cell):
    '''cells with bangs for commands i.e. !ls'''
    if not has_code(cell):
        return 0
    for line in cell['source']:
        if re.search('^\!', line):
            return 1
    return 0

def hash_cell(cell):
    # XXX add for other cell types
    if cell['cell_type'] != 'code':
        return 0
    # XXX make more resilient to trivial differences
    return hash(''.join(cell['source']))


def has_ordered_cells(notebook):
    '''were the code cells ordered in execution 1, 2, 3...?
    TODO: maybe only check for in linear order
    rather than checking if in a "published" format
    ex: 1, 6, 8 would pass, but doesn't currently
    '''
    i = 1
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        if i != cell['execution_count']:
            return 0
        i += 1
    return 1


# functions to execute per cell
CELLWISE = [
    probably_uses_pyspark,
    probably_uses_s3,
    draws_graph,
    has_traceback,
    lines,
    cell_magic,
    line_magic,
    line_bang
]

# functions to execute per notebook
NOTEBOOKWISE = [
    has_ordered_cells,
]


class NotebookRepositoryAnalyzer(object):
    # avoids replication if all notebooks are testing the same features
    # this should also include information like NOTEBOOKWISE & REPOSITORYWISE
    header = ['file_name', 'category', 'is_code'] + [f.__name__ for f in CELLWISE]

    def __init__(self, notebooks=None, **config):
        self.cellwise = CELLWISE
        self.notebookwise = NOTEBOOKWISE
        if config:
            self.__dict__.update(config)
        self.findings = {}
        if notebooks:
            map(self.analyze_notebook, notebooks)

    def analyze_notebook(self, notebook, root, name):
        # TEMP: category is based on what you have downloaded
        # ex: downloading machine learning notebooks or
        # a way in which to organize directories on local disk
        try:
            category = root.split('/')[1]
        except IndexError:
            category = 'local'
        try:
            if notebook['nbformat'] != 4:
                return
        except KeyError:
            return

        self.findings['notebooks'] = self.findings.get('notebooks', [])
        cells = defaultdict(float)
        for cell in notebook['cells']:
            if is_code(cell):
                cells[0] += 1.0
                for i, f in enumerate(self.cellwise):
                    index = i + 1
                    cells[index] += f(cell)
            # TODO: add markdown
            # if is_markdown(cell):
            #     cells[1] += 1.0
        # store the notebook name
        result = [name, category]
        width = range(len(cells))
        height = len(notebook['cells'])
        for i in width:
            result.append(round(cells[i]/height, 2))
        self.findings['notebooks'].append(result)

    def finish_analysis(self):
        return self.findings # XXX

        for f in self.repositorywise:
            self.findings[f.__name__] = f(self.findings)

        # write findings out to a csv to be used
        # by a machine learning algorithm for
        # further analysis etc.

        # labels are the header,
        # but with the addition of filenames
        # did we loose the notebook?
        return self.findings


def analyze_tree(path, output):
    an = NotebookRepositoryAnalyzer()
    for root, dirnames, filenames in os.walk(path):
        for basename in filenames:
            # skip checkpoint files
            # skip non-notebook files
            if basename.endswith('checkpoint.ipynb')\
            or not basename.endswith('.ipynb'):
                continue
            try:
                notebook = json.load(open(os.path.join(root, basename)))
            except ValueError as e:
                # something is wrong with the notebook's json
                # this happens when there is a merge conflict etc.
                continue
            an.analyze_notebook(notebook, root, basename)
    analysis_dict = an.finish_analysis()
    with open(output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        header = NotebookRepositoryAnalyzer.header

        writer.writerow(header)

        for notebook_features in analysis_dict['notebooks']:
            # each result of feature extraction should probably be
            # stored in a dictionary w/ feature name as key
            # this code assumes that the order of the header
            writer.writerow(notebook_features)

    # writes to csv and no longer returns
    # return an.finish_analysis()

if __name__ == '__main__':
    import sys
    directory = sys.argv[1]
    output = sys.argv[2]
    analyze_tree(directory, output)
