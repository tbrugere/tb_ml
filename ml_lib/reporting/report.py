from typing import Literal

import itertools as it
import builtins
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from datetime import datetime
import glob

import numpy as np

from ..misc import auto_repr
from ..experiment_tracking import Model as DatabaseModel, Checkpoint as DatabaseCheckpoint, Test as DatabaseTest

@auto_repr("type", "path", "title")
class Row:
    #If I code something like  this again, 
    # probably inheritance would be better than a "type" variable 
    # and a lot of match-case on it
    # more extensible too
    type: Literal["config", "statistics", "recon_loss"]
    path: list[str]
    title: str|None = None
    def __init__(self, s, title=None):
        type, *path = s.strip().split("/")
        self.type = type
        self.path = path
        """
        Path is the following
        config/path_in_config

        statistic/stat_name/ot -> for statistic optimal transport error
        statistic/stat_name/recon -> for statistic recon error (directly compared with gt)

        statistic/distance_name/inter -> for OT using inter-set distance 
        statistic/distance_name/intra -> not useful here but for OT using intra-set distance
        statistic/distance_name/recon -> for 1-to-1 distances
        """
        self.title = title

    def html_title(self) -> str:
        match self.type:
            case _ if self.title is not None:
                return self.title
            case "recon_loss":
                return "Reconstruction loss"
            case "statistics":
                metric_name, *metric_type = self.path
                return metric_name
        return self.path[-1]

    def get(self, model_config, model_statistics, recon_losses, gt_statistics, is_val=False):
        if isinstance(gt_statistics, dict) :
            if model_config is not None: gt_statistics = gt_statistics[model_config.datasets.val.parent]
            elif len(gt_statistics) == 1: gt_statistics, = gt_statistics.values()
        match self.type:
            case "config":
                if is_val: return "validation set"
                if model_config is None: return None
                return self.get_config(model_config)
            case "statistics":
                if model_statistics is None: return None
                return self.get_statistic(model_statistics, gt_statistics, is_val=is_val)
            case "recon_loss":
                if recon_losses is None: return None
                return np.mean(recon_losses)
            case _:
                raise ValueError(f"wrong type for row {self.type}")

    def get_config(self, model_config):
        for p in self.path:
            model_config = model_config[p]
        return model_config

    def get_statistic(self, model_statistics, gt_statistics, is_val=False):
        return model_statistics[self.path[0]]


    def get_full_row(self, models_configs, model_statistics, recon_losses, gt_statistics, val_index=None):
        values = [self.get(conf, stats, recon_loss, gt_statistics, i == val_index)
                 for conf, stats, recon_loss, i in zip(models_configs, model_statistics, recon_losses, it.count())]
        return values

#Row("config/model/type").get_full_row(configs, statistics, statistics_gt)

class Table():
    rows = []
    gt_column=False
    """doesnt do anything yet"""

    val_column: bool|int = False
    """True if the last column is evaluation metrics
    False if no evaluation metrics
    int if the column is at a specific index
    """


    css = """
    .table-container {
        overflow-x: auto;
        witdh: 100%;
    }
    table{
        width: auto;
    }
    @media print {
        table {
            width: 100%;
            overflow-wrap: break-word;
        }
    }   

    tbody {
        border-top: 1px solid black
    }
    tbody:first-child {
        border-top: none
    }
    """

    def __init__(self, rows, gt_column=False, val_column=False):
        def convert_to_row(r):
            match r:
                case Row():
                    return r
                case "---" | "..." | builtins.Ellipsis: 
                    return ...
                case str(s):
                    return Row(s)
                case [*rows]:
                    return [convert_to_row(row) for row in rows]
                    
        rows = [convert_to_row(row) for row in rows]
        self.rows = rows
        self.gt_column = gt_column
        self.val_column = val_column

    def get_full_row_html(self, row_title, row_values,all_th=False, stringio = None, precision=2, bold_lowest=False):
        td = "td" if not all_th else "th"
        s = stringio if stringio is not None else StringIO()
        s.write("<tr>")
        s.write(f"<th>{row_title}<th>")
        if bold_lowest:
            row_values_with_infty = [v if v is not None else np.inf for v in row_values]
            match self.val_column:
                case None:
                    pass
                case True:
                    row_values_with_infty[-1] = np.inf
                case int(val_column):
                    row_values_with_infty[val_column] = np.inf
            lowest_index = np.argmin(row_values_with_infty)
        else: lowest_index = -1
        for i, value in enumerate(row_values):
            match value:
                case None:
                    s.write(f"<{td}>∅</{td}>")
                case float() if i == lowest_index:
                    s.write(f"<{td}><b>{value:.{precision}}</b></{td}>")
                case float():
                    s.write(f"<{td}>{value:.{precision}}</{td}>")
                case _:
                    s.write(f"<{td}>{value}</{td}>")
        s.write("</tr>")
        if not stringio:
            return s.getvalue()


    def get_html(self, models_configs, model_statistics, recon_losses, gt_statistics, 
                 first_line_is_header=True):
        
        match self.val_column:
            case True: val_index = len(model_statistics) - 1
            case False: val_index = None
            case int(i): val_index = i
            case _: raise ValueError(f"wrong val_column value {self.val_column}")
        #--------------------------------------- utility functions
        def close_current_env(current_env, s:StringIO):
            match current_env:
                case "<thead>":
                    s.write("</thead>")
                case "<tbody>":
                    s.write("</tbody>")

        #--------------------------------------- 
        s = StringIO()
        s.write(f"<style>{self.css}</style>")
        s.write("<div class='table-container'><table>")
        current_env = "<thead>" if first_line_is_header else "<tbody>"
        s.write(current_env)
        for i, row in enumerate(self.rows):
            all_th = first_line_is_header and i == 0
            if row == ...: # change table subdivision
                close_current_env(current_env, s)
                current_env = "<tbody>"
                s.write(current_env)
                continue
            if not isinstance(row, Sequence):
                row = (row,)
            values_list =[
                r.get_full_row(models_configs, 
                               model_statistics,recon_losses, gt_statistics, val_index=val_index)
                for r in row]
            values = self.merge_values(values_list)
            title = "/ ".join(r.html_title() for r in row)
            bold_lowest = row[0].type == "statistics"
            self.get_full_row_html(title, values,
                all_th=all_th, stringio=s, bold_lowest=bold_lowest)
        close_current_env(current_env, s)
        s.write("</table></div>")
        return s.getvalue()

    def get_jupyter_html(self, models_configs, model_statistics, gt_statistics, 
                 first_line_is_header=True):
        from IPython.core.display import HTML
        return HTML(self.get_html(models_configs, model_statistics, gt_statistics, 
                 first_line_is_header))
    @staticmethod
    def merge_values(values_list):
        """Takes a list of rows values list (so list of list of values) [v0 .. vn]
        with potential None values,
        and returns a single values list
        v[i] = v0[i] if v0[i] is not None else v1[i] if v1[i] is not None else ... else None
        """
        values0, *rest_values = values_list
        v = values0
        for other_values in rest_values:
            for i in range(len(v)):
                if v[i] is None:
                    v[i] = other_values[i]
        return v
    
    
def generate_report(*, 
                    include_graphs:bool = True,
                    include_examples:bool = True,
                    config_paths: list[Path|str] = [], 
                    output: Path|None = None, 
                    pipeline_dir: Path|None = None,
                    format: Literal["pdf", "html"]|None=None, 
                    workdir: Path = Path(".")):
    from papermill import execute_notebook
    import nbformat
    from nbconvert import WebPDFExporter, HTMLExporter 
    from nbconvert.preprocessors import TagRemovePreprocessor
    intermediate_notebook = workdir / "Output.ipynb"

    if format is None and output is not None:
        format = "html" if output.suffix == ".html" else "pdf"
    elif format is None:
        format = "pdf"

    # Process arguments
    if output is None:
        report_name = f"report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{format}"
        output = workdir / "reports" / report_name

    if pipeline_dir is None:
        root_dir = Path(__file__).parent.parent.parent
        pipeline_dir = root_dir / "Pipeline" 

    #expand potential globs in config_paths
    new_config_paths = []
    for p in config_paths:
        if "*" in str(p):
            new_config_paths.extend(glob.glob(str(p)))
        else:
            new_config_paths.append(p)
    config_paths = new_config_paths


    execute_notebook(
        pipeline_dir / "Report.ipynb",
        intermediate_notebook, 
        dict(
            config_paths=[str(p) for p in config_paths]
        )
    )
    print(f"Report notebook ran in {intermediate_notebook}")

    intermediate_notebook_node = nbformat.read(intermediate_notebook, as_version=4)

    remove_processor = TagRemovePreprocessor()
    remove_tags = []
    if not include_graphs: remove_tags.append("statistics")
    if not include_examples: remove_tags.append("examples")
    remove_processor.remove_cell_tags = remove_tags

    if format == "html":
        exporter = HTMLExporter(exclude_output_prompt=True,
                                  exclude_input=True,
                                  exclude_input_prompt=True,
                                  )
    else:
        exporter = WebPDFExporter(exclude_output_prompt=True,
                                  exclude_input=True,
                                  exclude_input_prompt=True,
                                  allow_chromium_download=True)
    exporter.register_preprocessor(remove_processor, True)
    body, resources = exporter.from_notebook_node(intermediate_notebook_node)

    match body:
        case bytes(body):
            output.write_bytes(body)
        case str(body):
            output.write_text(body)
    print(f"Report written in {output}")

