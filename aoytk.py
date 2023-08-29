""" AOY-TK module. Provides functions and forms to simplify web-archive analysis. 
"""
# AOY-TK Module
import ipywidgets as widgets
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from IPython.display import clear_output
import spacy
from wordcloud import WordCloud
import random
from pathlib import Path

# Global path variable -- a default for Google Drive usage
# default path, can be overwritten by the path-setter widget
path = "/content/drive/MyDrive/AOY/"


# General purpose functions.
def display_path_select():
    """Displays a text box to set the default path for reading/writing data."""
    txt_path = widgets.Text(
        description="Folder path:",
        placeholder="Enter your folder path",
        value="/content/drive/MyDrive/AOY/",
    )

    def btn_set_path(btn):
        """On-click function for the path-set button."""
        global path
        if txt_path.value == "":
            print("Please enter a path value.")
        else:
            path = txt_path.value
            print(f"Folder path set to: {path}")

    btn_txt_submit = widgets.Button(description="Submit")
    btn_txt_submit.on_click(btn_set_path)
    display(txt_path)
    display(btn_txt_submit)


def get_files(self, file_types):
    """Recursively list files of given types from directory + its subdirectories.

    Looks for files in the working directory, specified by the global `path`
    variable.

    Args:
        file_types: Tuple of strings specifying the file types to match on.
            For example: (".csv", ".parquet", ".pqt")

    Returns:
        A list of the matching file names, including their path relative to the
        top level directory. I.e. a file in the top-level directory will only
        return the file name, a file in a sub-directory will include the
        sub-directory name and the file name. Ex. "subdir/a.csv"
    """
    matched_files = []
    for dirpath, subdirs, files in os.walk(path):
        subfolder = re.sub(path, "", dirpath)
        datafiles = [f for f in files if f.endswith(file_types)]
        for f in datafiles:
            if subfolder == "":
                matched_files.append(f)
            else:
                matched_files.append(f"{subfolder}/{f}")
    return matched_files


# Fletcher's code to download a WARC file from a direct link
def download_file(url, filepath="", filename=None, loud=True):
    """Downloads a file from the specified URL into the specified folder.

    Optionally specify a specific filename for the downloaded file. Optionally
    display download progress via a progress bar.

    Args:
        url: String specifying the URL path to download the file from
        filepath: The file path specifying the folder to save the file into
        filename: The filename to give to the downloaded file. If None, the
            filename will be extracted from the URL.
        loud: Boolean indicating whether or not to display download progress
    """
    if not filename:
        filename = url.split("/")[-1]
        if "?" in filename:
            filename = filename.split("?")[0]

    r = requests.get(url, stream=True)
    if loud:
        total_bytes_dl = 0
        content_len = int(r.headers["Content-Length"])
        prog_bar = widgets.IntProgress(
            value=1,
            min=0,
            max=100,
            step=1,
            bar_style="info",
            orientation="horizontal",
        )
        print(f"Download progress of {filename}:")
        display(prog_bar)

    with open(filepath + filename, "wb") as fd:
        for chunk in r.iter_content(chunk_size=4096):
            fd.write(chunk)
            if loud:
                total_bytes_dl += 4096
                percent = int((total_bytes_dl / content_len) * 100.0)
                prog_bar.value = percent
    r.close()
    print("File download completed.")


def display_download_file():
    """Display textbox to download file from specified URL."""
    txt_url = widgets.Text(description="W/ARC URL: ")
    btn_download = widgets.Button(description="Download W/ARC")

    def btn_download_action(btn):
        """On-click function to trigger the download of file using form input."""
        url = txt_url.value
        if url != "":
            download_file(
                url, path + "/"
            )  # download the file to the specified folder set in the above section
        else:
            print("Please specify a URL in the textbox above.")

    btn_download.on_click(btn_download_action)
    display(txt_url)
    display(btn_download)


class DerivativeGenerator:
    """Creates derivative files from W/ARCs.

    This class contains all of the functions relating to derivative generation.
    """

    def __init__(self):
        """Initialize the dependencies for creating derivatives."""
        # initialize the PySpark context
        import findspark

        findspark.init()
        import pyspark

        self.sc = pyspark.SparkContext()
        from pyspark.sql import SQLContext

        self.sqlContext = SQLContext(self.sc)

    def create_csv_with_header(self, headers, datafile, outputfile):
        """Create a version of data file with the specified headers.

        Args:
            headers: A list of column headers in the desired order
            datafile: The path of the CSV file, without headers, to add headers
                to
            outputfile: The path of the desired output file
        """
        # change the field size to accommodate large data
        import csv
        import sys

        maxInt = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)

        with open(outputfile, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headers)
            with open(datafile, "r") as datafile:
                reader = csv.reader(datafile, quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    writer.writerow(row)

    # this code is rather messy and could probably use refactoring at some point
    def generate_derivative(
        self,
        source_file,
        output_folder,
        file_type="csv",
        deriv_type="text",
        text_filters=0,
        file_category="images",
    ):
        """Create a text or file derivative file from the specified source file.

        Create a text or file-type derivative from the specified W/ARC source
        file, using the output settings specified.

        Args:
            source_file: Path to the W/ARC file to generate the derivative from
            output_folder: The name for the output folder to save the derivative
                into. Note: this is currently a relative path, the folder will be
                created as a sub-folder of the working folder, specified by the
                global `path` variable)
            file_type: The file format to save the produced derivative in.
                Can be either "csv" or "parquet".
            deriv_type: "text" for text derivatives or "file" for file type
                derivatives
            text_filters: An integer representing which type of text filtering
                to apply to the generated derivative. Only applies for
                deriv_type = "text".
                0 : return the complete text content of each webpage (with HTML
                    tags removed)
                1 : return the complete text with HTTP headers removed
                2 : return the text with the boilerplate removed (boilerplate
                    includes nav bars etc)
            file_category: a string representing the type of file derivative to
                be created. Supported types are: "audio", "images", "pdfs",
                "presentations", "spreadsheets", "videos", and "word processor"

        Raises:
          AnalysisException (from pyspark.sql.utils) if file path already exists
          Exception if anything else goes wrong and the "_SUCCESS" file is not
          created in the output directory.
        """

        # import the AUT (needs to be done after the PySpark set-up)
        from aut import (
            WebArchive,
            remove_html,
            remove_http_header,
            extract_boilerplate,
            extract_domain,
        )
        from pyspark.sql.functions import col, desc

        # create our WebArchive object from the W/ARC file
        archive = WebArchive(self.sc, self.sqlContext, source_file)

        if deriv_type == "text":
            if text_filters == 0:
                content = remove_html("content")
            elif text_filters == 1:
                content = remove_html(remove_http_header("content"))
            else:
                content = extract_boilerplate(remove_http_header("content")).alias(
                    "content"
                )

            archive.webpages().select(
                "crawl_date", "domain", "url", content
            ).write.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").format(
                file_type
            ).option(
                "escape", '"'
            ).option(
                "encoding", "utf-8"
            ).save(
                output_folder
            )

        elif deriv_type == "file":
            if file_category == "audio":
                # get the audio derivative -- following the format from AUT
                archive.audio().select(
                    "crawl_date",
                    extract_domain("url").alias("domain"),
                    "url",
                    "filename",
                    "extension",
                    "md5",
                    "sha1",
                ).write.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").format(
                    file_type
                ).option(
                    "escape", '"'
                ).option(
                    "encoding", "utf-8"
                ).save(
                    output_folder
                )

            elif file_category == "images":
                # get the image derivative -- following the format from AUT
                archive.images().select(
                    "crawl_date",
                    extract_domain("url").alias("domain"),
                    "url",
                    "filename",
                    "extension",
                    "width",
                    "height",
                    "md5",
                    "sha1",
                ).write.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").format(
                    file_type
                ).option(
                    "escape", '"'
                ).option(
                    "encoding", "utf-8"
                ).save(
                    output_folder
                )

            elif file_category == "pdfs":
                archive.pdfs().select(
                    "crawl_date",
                    extract_domain("url").alias("domain"),
                    "url",
                    "filename",
                    "extension",
                    "md5",
                    "sha1",
                ).write.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").format(
                    file_type
                ).option(
                    "escape", '"'
                ).option(
                    "encoding", "utf-8"
                ).save(
                    output_folder
                )

            elif file_category == "presentations":
                archive.presentation_program().select(
                    "crawl_date",
                    extract_domain("url").alias("domain"),
                    "url",
                    "filename",
                    "extension",
                    "md5",
                    "sha1",
                ).write.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").format(
                    file_type
                ).option(
                    "escape", '"'
                ).option(
                    "encoding", "utf-8"
                ).save(
                    output_folder
                )

            elif file_category == "spreadsheets":
                archive.spreadsheets().select(
                    "crawl_date",
                    extract_domain("url").alias("domain"),
                    "url",
                    "filename",
                    "extension",
                    "md5",
                    "sha1",
                ).write.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").format(
                    file_type
                ).option(
                    "escape", '"'
                ).option(
                    "encoding", "utf-8"
                ).save(
                    output_folder
                )

            elif file_category == "videos":
                archive.video().select(
                    "crawl_date",
                    extract_domain("url").alias("domain"),
                    "url",
                    "filename",
                    "extension",
                    "md5",
                    "sha1",
                ).write.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").format(
                    file_type
                ).option(
                    "escape", '"'
                ).option(
                    "encoding", "utf-8"
                ).save(
                    output_folder
                )

            elif file_category == "word processor":
                archive.word_processor().select(
                    "crawl_date",
                    extract_domain("url").alias("domain"),
                    "url",
                    "filename",
                    "extension",
                    "md5",
                    "sha1",
                ).write.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").format(
                    file_type
                ).option(
                    "escape", '"'
                ).option(
                    "encoding", "utf-8"
                ).save(
                    output_folder
                )

            else:
                print(f"Unsupported file category: {file_category}. ")
                return False

        # rename the datafile to have a meaningful title, remove the success file
        success = False
        # the folder will contain exactly 2 files, a _SUCCESS file and the resulting datafile
        for f in os.scandir(output_folder):
            if f.path.split("/")[-1] == "_SUCCESS":
                # indicate that the derivative was generated successfully
                success = True
                # remove the success indicator file
                os.remove(f.path)
            # for the datafile
            if f.path.split(".")[-1] == file_type:
                source_file_name = source_file.split(".")[0]
                source_file_name = source_file_name.split("/")[-1]
                # add the appropriate headers
                if file_type == "csv":
                    headers = []
                    # for all text_filters between 0 and 2, we'll use the same header
                    # if adding different derivatives, add the appropriate headers here!
                    if deriv_type == "text" and text_filters >= 0 and text_filters <= 2:
                        headers = ["crawl_date", "domain", "url", "content"]
                    elif deriv_type == "file":
                        if file_category == "images":
                            headers = [
                                "crawl_date",
                                "domain",
                                "url",
                                "filename",
                                "extension",
                                "width",
                                "height",
                                "md5",
                                "sha1",
                            ]
                        else:
                            headers = [
                                "crawl_date",
                                "domain",
                                "url",
                                "filename",
                                "extension",
                                "md5",
                                "sha1",
                            ]
                    output_path = output_folder + source_file_name + ".csv"
                    self.create_csv_with_header(headers, f.path, output_path)
                    os.remove(f.path)
                else:
                    os.rename(
                        f.path,
                        output_folder + source_file_name + "." + file_type,
                    )
        if not success:
            raise Exception()

    def display_derivative_creation_options(self):
        """Displays a form to set options for derivative file creation.

        Displays 4 form elements to select:
        - any W/ARC file from within the defined working folder to create a
          derivative of
        - desired type of derivative (i.e. what content to include in the
          derivative)
        - the output folder for the derivative (will be created within the
          working directory)
        - the desired output file type (csv or parquet)

        Also displays a button which, on-click, will run generate_derivative(),
        passing in the settings specified in the form.
        """
        # exception class for PySpark -- thrown often when path is incorrect
        from pyspark.sql.utils import AnalysisException

        # file picker for W/ARC files in the specified folder
        data_files = get_files(path, (".warc", ".arc", "warc.gz", ".arc.gz"))
        file_options = widgets.Dropdown(description="W/ARC file", options=data_files)
        out_text = widgets.Text(description="Output folder", value="output/")
        format_choice = widgets.Dropdown(
            description="Output format", options=["csv", "parquet"], value="csv"
        )
        # text content choices
        content_options = [
            "All text content",
            "Text content without HTTP headers",
            "Text content without boilerplate",
        ]
        content_choice = widgets.Dropdown(
            description="Content", options=content_options
        )
        content_val = content_options.index(content_choice.value)
        text_button = widgets.Button(description="Create derivative")

        # text deriv options layout
        text_deriv_layout = widgets.VBox(
            [file_options, out_text, format_choice, content_choice, text_button]
        )

        file_content_options = [
            "audio",
            "images",
            "pdfs",
            "presentations",
            "spreadsheets",
            "videos",
            "word processor",
        ]
        file_content_choice = widgets.Dropdown(
            description="File types", options=file_content_options
        )
        file_button = widgets.Button(description="Create derivative")

        file_deriv_layout = widgets.VBox(
            [
                file_options,
                out_text,
                format_choice,
                file_content_choice,
                file_button,
            ]
        )

        # this function is defined here in order to keep the other form elements
        # in-scope and therefore allow for the reading of their values
        def btn_create_text_deriv(btn):
            """On-click function for the create derivative button.

            Retrieves the values from the other inputs on the form and passes
            them to generate_derivative() to create a derivative file using the
            selected settings.
            """
            content_options = [
                "All text content",
                "Text content without HTTP headers",
                "Text content without boilerplate",
            ]
            input_file = path + "/" + file_options.value
            output_location = path + "/" + out_text.value
            content_val = content_options.index(content_choice.value)
            print("Creating derivative file... (this may take several minutes)")

            try:
                self.generate_derivative(
                    input_file,
                    output_location,
                    file_type=format_choice.value,
                    deriv_type="text",
                    text_filters=content_val,
                )
                print("Derivative generated, saved to: " + output_location)
            except AnalysisException as e:
                # ensure the analysis exception is thrown because the path exists
                # (update if a better method of checking is found)
                if "path" in str(e) and "already exists" in str(e):
                    print(
                        "Specified output folder already exists. Please specify a"
                        " new output folder name."
                    )
                else:
                    print(e)
            except Exception as e:
                print(
                    "An error occurred while processing the W/ARC. Derivative "
                    "file may not have been generated successfully."
                )
                print(f"Error message: {e}")

        def btn_create_file_deriv(btn):
            """On-click function for the create file derivative button.

            Retrieves the values from the other inputs on the form and passes
            them to generate_derivative() to create a derivative file using the
            selected settings.
            """
            input_file = path + "/" + file_options.value
            output_location = path + "/" + out_text.value

            print("Creating derivative file... (this may take several minutes)")
            try:
                self.generate_derivative(
                    input_file,
                    output_location,
                    file_type=format_choice.value,
                    deriv_type="file",
                    file_category=file_content_choice.value,
                )
                print("Derivative generated, saved to: " + output_location)
            except AnalysisException as e:
                # ensure the analysis exception is thrown because the path exists
                # (update if a better method of checking is found)
                if "path" in str(e) and "already exists" in str(e):
                    print(
                        "Specified output folder already exists. Please specify a"
                        " new output folder name."
                    )
                else:
                    print(e)
            except Exception as e:
                print(
                    "An error occurred while processing the W/ARC. Derivative "
                    "file may not have been generated successfully."
                )
                print(f"Error message: {e}")

        # assign
        text_button.on_click(btn_create_text_deriv)
        file_button.on_click(btn_create_file_deriv)

        # make the tab display
        tab = widgets.Tab(children=[text_deriv_layout, file_deriv_layout])
        [
            tab.set_title(i, title)
            for i, title in enumerate(["Text Derivatives", "File Type Derivatives"])
        ]
        display(tab)


class Analyzer:

    """Tools for analyzing W/ARC derivatives."""

    def __init__(self):
        """Initializes the Analyzer class. Sets attributes to default values."""
        self.data = None
        self.data_loaded = False
        self.number_LDA_Topics = None

        # change display settings for pandas to show more content
        pd.set_option("display.max_colwidth", 100)
        # pd.set_option('display.max_columns', None)
        pd.set_option("display.max_rows", 200)

    def set_data(self, datafile):
        """Sets the data attribute for the Analyzer.

        Parses columns to appropriate types if applicable.

        Args:
            datafile: String specifying the path to the data file to analyze.
        """
        self.data = pd.read_csv(datafile)
        self.data_loaded = True

        # if the crawl_date column is included on the frame, make it a date
        if "crawl_date" in list(self.data):
            # Currently, pandas is not doing a great job of auto-detecting the date
            # format correctly. Temporarily, this code is included to correctly
            # detect and parse the date formats that we have tested on.
            # This process should be made more robust in the future.

            # get the first date, convert to str
            date = str(self.data["crawl_date"].iloc[0])
            if re.match(r"^[0-9]+$", date):
                if len(date) == 14:  # format='%Y%m%d%H%M%S'
                    self.data["crawl_date"] = pd.to_datetime(
                        self.data["crawl_date"], format="%Y%m%d%H%M%S"
                    )
                elif len(date) == 8:  # format='%Y%m%d'
                    self.data["crawl_date"] = pd.to_datetime(
                        self.data["crawl_date"], format="%Y%m%d"
                    )
                else:  # hope that pandas figures it out
                    self.data["crawl_date"] = pd.to_datetime(self.data["crawl_date"])
            else:
                self.data["crawl_date"] = pd.to_datetime(self.data["crawl_date"])

    def load_data(self):
        """Displays a form allowing the user to load a data file to work with.

        Allows the user to select any .csv file that is in the current working
        directory (specified by the `path` variable), or any of its
        sub-directories.
        """
        # display the options available in the working directory
        # Parquet files are not currently supported, if/when they are, add
        # '".parquet", ".pqt"' to the file ending options
        label = widgets.Label("Derivative file to analyze ")
        file_options = widgets.Dropdown(
            description="", options=get_files(path, (".csv"))
        )
        button = widgets.Button(description="Select file")

        def btn_select_file(btn):
            """Accepts form input and tries to load the selected file."""
            selected_file = path + "/" + file_options.value
            print("Loading data...")
            self.set_data(selected_file)
            print(f"Data loaded from: {selected_file}")

        button.on_click(btn_select_file)
        display(widgets.HBox([label, file_options]))
        display(button)

    def date_range_select(self):
        """Create a date range selector for valid dates in the data."""
        from IPython.display import display, Javascript

        valid_range = self.data.reset_index()["crawl_date"].agg(["min", "max"])
        start_label = widgets.Label("Select a start date  ")
        start_picker = widgets.DatePicker(
            description="", value=valid_range["min"].date(), disabled=False
        )
        start_picker.add_class("start-date")
        end_label = widgets.Label("Select an end date ")
        end_picker = widgets.DatePicker(
            description="", value=valid_range["max"].date(), disabled=False
        )
        end_picker.add_class("end-date")

        js = f"""const query = '.start-date > input:first-of-type';
           document.querySelector(query).setAttribute('min', '{valid_range['min'].strftime('%Y-%m-%d')}');
           document.querySelector(query).setAttribute('max', '{valid_range['max'].strftime('%Y-%m-%d')}'); 
           const q = '.end-date > input:first-of-type';
           document.querySelector(q).setAttribute('min', '{valid_range['min'].strftime('%Y-%m-%d')}');
           document.querySelector(q).setAttribute('max', '{valid_range['max'].strftime('%Y-%m-%d')}');"""
        script = Javascript(js)

        # try returning the widgets for use / display in various places
        # returns a list containing all of the widgets, the labels and selectors
        # are paired in tuples, the JS for the calendar is seperate
        return [(start_label, start_picker), (end_label, end_picker), script]

    def display_top_domains(self):
        """Display the most frequently crawled domains in the dataset.

        Number of domains displayed is controlled using an IntSlider.
        """
        domain_values = self.data["domain"].value_counts()
        n_domains = len(domain_values)

        def top_domains(n):
            """Prints the n top domains."""
            print(domain_values.head(n))

        n_slider = widgets.IntSlider(value=10, max=n_domains)
        out = widgets.interactive_output(top_domains, {"n": n_slider})
        print(f"There are {n_domains} different domains in the dataset. ")
        display(n_slider)
        display(out)

    def plot_3d_crawl_frequency(self, aggregated_crawl_count):
        """Creates a 3D plot of the crawl frequency in the passed dataframe.

        Args:
            aggregated_crawl_count: A pandas dataframe containing the domains of
                interest to be plotted, with their crawl counts aggregated by
                some frequency
        """
        from matplotlib.collections import PolyCollection

        # first get the crawl dates for the axis labels
        crawl_dates = sorted(
            set(
                aggregated_crawl_count.index.get_level_values(
                    "crawl_date"
                ).to_pydatetime()
            )
        )
        cd_to_xtick = {d: i for i, d in enumerate(crawl_dates)}
        reverse_cd_map = {i: d for i, d in enumerate(crawl_dates)}
        # domain names by number of crawls, least -> greatest
        domains_by_num_crawls = (
            aggregated_crawl_count.groupby(level=0).sum().url.sort_values().index
        )

        verts = []
        last_domain = None
        max_crawl_count = 0

        for d in domains_by_num_crawls:
            current_polygon = []
            for t in aggregated_crawl_count.loc[d].sort_index().itertuples():
                tstamp = t.Index.to_pydatetime()
                tstamp_int = cd_to_xtick[tstamp]
                crawl_count = t.url
                max_crawl_count = max(crawl_count, max_crawl_count)
                if not current_polygon:  # for the first polygon
                    current_polygon.append((tstamp_int, 0))
                current_polygon.append((tstamp_int, crawl_count))
            current_polygon.append((current_polygon[-1][0], 0))
            verts.append(current_polygon)

        # now that we have the polygons, set up the plot itself
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(18, 24), subplot_kw={"projection": "3d"}
        )
        facecolors = plt.get_cmap("twilight_shifted_r")(np.linspace(0, 1, len(verts)))
        # testing out even spacing of the polygons
        zs = range(0, len(verts) * 2, 2)

        poly = PolyCollection(verts, facecolors=facecolors, alpha=0.7)
        ax1.add_collection3d(poly, zs=zs, zdir="x")

        max_y = max(reverse_cd_map)
        ax1.set(
            xlim=(0, max(zs) + 2),
            ylim=(-1, max_y + 1),
            zlim=(0, max_crawl_count),
            xlabel="",
            ylabel="",
            zlabel="crawls",
        )
        ax1.invert_xaxis()

        ax1.set_xticks(zs)
        ax1.set_xticklabels(domains_by_num_crawls, rotation=40, ha="right")

        ytick_labels = [
            reverse_cd_map[i].strftime("%Y-%m-%d") for i in sorted(reverse_cd_map)
        ]
        ax1.set_yticklabels(ytick_labels, rotation=-20, ha="left")
        ax1.set_yticks(range(max_y + 1))

        # 2nd plot, a difference angle than the first
        poly = PolyCollection(verts, facecolors=facecolors, alpha=0.7)
        ax2.add_collection3d(poly, zs=zs, zdir="y")

        max_x = max(reverse_cd_map)
        ax2.set(
            xlim=(-1, max_x + 1),
            ylim=(0, max(zs) + 2),
            zlim=(0, max_crawl_count),
            xlabel="",
            ylabel="",
            zlabel="crawls",
        )
        ax2.invert_xaxis()

        ax2.set_xticks(range(max_x + 1))
        x_tick_labels = [
            reverse_cd_map[i].strftime("%Y-%m-%d") for i in sorted(reverse_cd_map)
        ]
        ax2.set_xticklabels(x_tick_labels, rotation=40, ha="right")
        ax2.set_yticks(zs)
        ax2.set_yticklabels(domains_by_num_crawls, rotation=-20, ha="left")

    def plot_2d_crawl_frequency(self, aggregated_crawl_count, inflation_factor=2.5):
        """Creates a 2D plot of the crawl frequency for the given dataframe.

        Args:
            aggregated_crawl_count: a pandas dataframe containing the domains of
                interest to be plotted, with their crawl counts aggregated by
                some frequency
            inflation_factor: an optional float that changes the circle sizes on
                the plot
        """

        import math

        fig, ax = plt.subplots(figsize=(18, 12))
        crawl_dates = sorted(
            set(
                aggregated_crawl_count.index.get_level_values(
                    "crawl_date"
                ).to_pydatetime()
            )
        )
        domains_by_ncrawls = (
            aggregated_crawl_count.groupby(level=0).sum().url.sort_values().index
        )

        max_crawl_count = 0
        y = -1
        for d in domains_by_ncrawls:
            xaxis = []
            zaxis = []
            y = y + 1

            for t in aggregated_crawl_count.loc[d].sort_index().itertuples():
                tstamp = t.Index.to_pydatetime()
                crawl_count = t.url
                max_crawl_count = max(crawl_count, max_crawl_count)

                xaxis.append(tstamp)
                zaxis.append(
                    crawl_count**inflation_factor
                )  # artificially inflate size to create larger circles
            scatter = ax.scatter(xaxis, [d] * len(xaxis), s=zaxis)

        # Here we create a legend:
        # we'll plot empty lists with the desired size and label
        sizes = sorted(
            set(
                map(
                    lambda t: t.url**inflation_factor,
                    aggregated_crawl_count.loc[domains_by_ncrawls].itertuples(),
                )
            )
        )
        if 0 in sizes:
            sizes.remove(0)
        for area in sizes:
            ax.scatter(
                [],
                [],
                color=scatter.cmap(0.7),
                s=area,
                label=str(math.ceil(area ** (1 / inflation_factor))),
            )
        ax.legend(
            scatterpoints=1,
            loc="upper left",
            bbox_to_anchor=(1.025, 1),
            fancybox=True,
            frameon=True,
            shadow=True,
            handleheight=2.2,
            borderaxespad=0.0,
            borderpad=1,
            labelspacing=3.5,
            handlelength=4,
            handletextpad=3,
            title="Crawl count by size",
        )

    def create_crawl_frequency_graph(
        self, n, graph_type, freq="1M", start_date=None, end_date=None
    ):
        """Plots the crawl frequency of the top n domains in the dataset.

        Args:
            n: The number of the top domains to plot
            freq: String indicating the frequency to aggregate the data by.
                "1M" aggregates it in 1 month groups, "1W" in 1 week groups.
                A full list of frequencies is available:
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            graph_type: either '2d' for a 2-dimensional visualization of the crawl
                frequency, or '3d' for a 3-dimensional visualization of the crawl
                frequency.
            start_date: an optional string of the form 'YYYY-MM-DD' representing
                the first date of interest in the dataset. If this is not
                provided, the earliest date in the data will be selected.
            end_date: an optional string of the form 'YYYY-MM-DD' representing
                the last date of interest in the dataset. If this is not
                provided, the latest date in the data will be selected.
        """
        # this is a temporary line to ignore warning output for the demo video
        import warnings

        warnings.filterwarnings("ignore")
        # if no start date is specified, select the min date
        if start_date == None:
            start_date = self.data["crawl_date"].min()
        # similiarly for the end date
        if end_date == None:
            end_date = self.data["crawl_date"].max()

        date_mask = (self.data["crawl_date"] >= start_date) & (
            self.data["crawl_date"] <= end_date
        )
        subset_data = self.data.loc[date_mask]
        # aggregate the data
        domains = set(
            subset_data.groupby("domain")
            .count()
            .sort_values(by="crawl_date", ascending=False)
            .head(n)
            .index
        )
        frames = []
        for d in domains:
            # for each domain, get the count of all crawls in each month
            frames.append(
                subset_data.loc[self.data["domain"] == d]
                .groupby(pd.Grouper(key="crawl_date", freq=freq))
                .count()
            )
        aggregated_crawl_count = pd.concat(frames, keys=list(domains))

        # create the appropriate graph
        if graph_type == "3D":
            self.plot_3d_crawl_frequency(aggregated_crawl_count)
        elif graph_type == "2D":
            self.plot_2d_crawl_frequency(aggregated_crawl_count)
        else:
            print(
                f"{graph_type} is not a supported graph type. "
                "Please choose either '2D' or '3D'."
            )

        return aggregated_crawl_count

    def display_crawl_frequency(self):
        """Displays the interface for creating crawl frequency graphs."""

        graph_options = widgets.RadioButtons(
            options=["3D", "2D"], value="3D", disabled=False  # default to '3d'
        )
        graph_options_label = widgets.Label("Style of visualization: ")

        date_components = self.date_range_select()

        num_domains = widgets.IntSlider(
            value=10,
            min=1,
            max=12,
            step=1,
        )
        num_domains_label = widgets.Label("Number of domains: ")

        freq_options = widgets.Dropdown(
            options=[("Monthly", "1M"), ("Weekly", "1W"), ("Daily", "1D")],
            value="1M",
        )
        freq_label = widgets.Label("Time scale")

        def create_crawl_btn(btn):
            """Accepts form input and triggers creation of crawl freq graph."""
            clear_output(True)  # only clear the output when the new output is ready
            display_options()
            print("Creating visualization... ")
            self.create_crawl_frequency_graph(
                num_domains.value,
                graph_options.value,
                freq_options.value,
                start_date=pd.Timestamp(date_components[0][1].value),
                end_date=pd.Timestamp(date_components[1][1].value),
            )

        enter_button = widgets.Button(
            description="Create visualization",
        )
        enter_button.on_click(create_crawl_btn)

        def display_options():
            """Displays the output options.

            Allows for them to be redisplayed once the window is cleared to
            display a new graph.
            """
            # display the number of domains
            display(widgets.HBox([num_domains_label, num_domains]))

            # display the options
            display(widgets.HBox([graph_options_label, graph_options]))

            # format the date / frequency selectors as a grid to make things a little tidier
            time_labels = widgets.VBox(
                [date_components[0][0], date_components[1][0], freq_label]
            )
            time_selectors = widgets.VBox(
                [date_components[0][1], date_components[1][1], freq_options]
            )
            time_controls = widgets.HBox([time_labels, time_selectors])
            display(time_controls)
            display(date_components[2])
            # display the enter button -- maybe make this interactable and then not require the button later
            display(enter_button)

        display_options()

    def display_search(self, field="url"):
        """Allows searching of the dataframe for rows containing specified string.

        Args:
            field: The label of a str-type field to search. By default, this is
                the URL field, other string type fields, like the domain field
                or the text field could be used instead.
        """
        if self.data_loaded != True:
            print("Please load data first using the 'load_data()' function. ")
            return
        # note: only works with "str" type fields
        # remove pages with no text content
        df = self.data[self.data["content"].notnull()]

        def reduce(q, field):
            subset = df.loc[lambda d: d[field].str.contains(q)]
            display(subset)

        description = widgets.Label(
            f"Search for entries with '{field}' containing" " the following: "
        )
        q = widgets.Text()
        out = widgets.interactive_output(
            reduce, {"q": q, "field": widgets.fixed(field)}
        )

        display(widgets.VBox([description, q, out]))

    # Word Cloud Functions // Text Analysis Helpers
    def get_document_token_list(
        self,
        doc,
        lemmatize=False,
        remove_punct=False,
        remove_stop=False,
        lowercase=True,
    ):
        """Given a spaCy doc, return it as a list of string tokens.
        Args:
            doc: The spaCy doc object produced by processing the document through
                the nlp pipeline
            lemmatize: Boolean, whether or not to lemmatize tokens (convert them
                to their base word ex. buying, bought, buys -> buy)
            remove_punct: Boolean, whether or not to remove punctuation like '.'
                ',' etc. Should not affect punctuation like COVID-19)
            remove_stop: Boolean, whether or not to remove stop words. Uses
            spaCy's stopword list.
            lower: Boolean, whether or not to convert all tokens to lowercase
        Returns:
            tokens: The list of the string tokens from the document
            id: The document id, from the doc._.text_id attribute of the spaCy Doc
        """
        id = doc._.text_id  # save the id

        if remove_stop:
            doc = [token for token in doc if not token.is_stop]

        if remove_punct:
            doc = [token for token in doc if not token.is_punct]

        if lemmatize:
            doc = [token.lemma_ for token in doc]
            if lowercase:
                doc = [token.lower() for token in doc]
        else:
            if lowercase:
                doc = [token.lower_ for token in doc]

        # ensure that the tokens are converted to strings before returning
        # necessary if not lemmatizing or lowering
        for token in doc:
            if not isinstance(token, str):
                token = token.text

        return doc, id

        # if not lemmatize:
        #     if remove_punct:
        #         tokens = [token for token in doc if not token.is_punct]
        #     else:
        #         tokens = [token for token in doc]
        # else:
        #     if remove_punct:
        #         tokens = [token.lemma_ for token in doc if not token.is_punct]
        #     else:
        #         tokens = [token.lemma_ for token in doc]
        # if lowercase:
        #     tokens = [token.lower_ for token in tokens]
        # else:
        #     tokens = [token for token in tokens]
        # return tokens, id

    # ideally, the stopwords parameter would let you specify what stopwords list
    # to use, for now it's a yes/no for using spaCy's stopwords list
    def preprocess_content(
        self,
        dataframe,
        lemmatize=False,
        remove_punct=False,
        remove_stop=False,
        lowercase=True,
    ):
        """Preprocesses the webpage data from the provided dataframe.

        Updates dataframe with additional information for each entry (length,
        readability scores).
        Returns tokenized content for further assessment/evaluation
        Args:
            dataframe: DataFrame containing all relevant information for
                scrapes of a given URL
            Preprocessing option arguments:
                lemmatize: Boolean, whether or not to apply lemmatization to
                    the text of the webpages
                remove_punct: Boolean, whether or not to remove punctuation
                    (like '.' ',' etc. should not affect punctuation like
                    COVID-19)
                remove_stop: Boolean, whether or not to remove stop words,
                    uses spaCy's stopword list.
                lowercase: Boolean, whether or not to convert tokens to
                    lowercase
        Returns:
            dataframe: A new pandas dataframe with the added information
            token_docs: Tokenized documents as a list of tuples of the form
                [([document_tokens], document_id)] where [document_tokens] is
                a list of spaCy token objects.

        """
        nlp = spacy.load("en_core_web_md")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        # ensure documents contain the id of their dataframe row so they can be
        # matched up later
        from spacy.tokens import Doc

        # if there is no registered extension, set one
        if not Doc.has_extension("text_id"):
            Doc.set_extension("text_id", default=None)

        # create the text tuples to add the text_id to the documents
        text_tuples = []
        for id, text in dataframe.content.items():
            text_tuples.append((text, id))

        # process the tuples using nlp.pipe
        # (disabling the components that we don't currently use to save time)
        # we may choose to use tok2vec at some point in the future, at which
        # point we can reenable it
        doc_tuples = nlp.pipe(
            text_tuples, as_tuples=True, disable=["tok2vec", "parser"]
        )

        docs = []  # list of all documents, including their ids in ._.text_id
        for doc, id in doc_tuples:
            doc._.text_id = id
            docs.append(doc)

        # extract lists of tokens
        token_docs = []
        for doc in docs:
            # bundle the text ids with the documents
            token_doc, id = self.get_document_token_list(
                doc,
                lemmatize=lemmatize,
                remove_punct=remove_punct,
                remove_stop=remove_stop,
                lowercase=lowercase,
            )
            token_docs.append((token_doc, id))
        return dataframe, token_docs

    def get_token_freq(self, token_docs, data):
        """Creates a dictionary of dictionaries with the token frequency.

        Args:
            token_docs: the list of (tokenized_document, text_id) tuples to
                process
            data: the dataframe from which the token docs were generated
        Returns:
            doc_freqs: dictionary of dictionaries where the key to the outer
                dictionary is the text id and the inner dictionary is
                token_string : frequency pairs {id: {token_string : frequency}}
        """
        doc_freqs = dict()
        for doc, id in token_docs:
            freqs = dict()
            for token in doc:
                t = token.lower()
                if t in freqs.keys():
                    freqs[t] += 1
                else:
                    freqs[t] = 1
            # this is probably a very strange way of doing this, but for
            # consistent access, this sets the date as the key for the
            # dictionary
            # NOTE: if we have a dataset with multiple scrapes in a day, this is
            #  not a valid technique and would need to be amended
            doc_freqs[str(data.at[id, "crawl_date"]).split(" ")[0]] = freqs
        return doc_freqs

    def identity_tokenizer(self, text):
        """Returns the input unchanged."""
        return text

    def get_tfidf_vectors(self, tokenized_docs_list, base_dataframe):
        """Calculates the TF-IDF matrix for the provided list of documents.

        Args:
            tokenized_docs_list: a list of lists of the form [([tokens], id)] where
                [tokens] is the list of string tokens in the document
                id is the text_id of the document that can be used to match it up
                    with its original entry in the dataframe
            base_dataframe: this is the pandas dataframe that we are matching our
                doc ids back to
        Returns:
            df: A new pandas dataframe representing the TF-IDF matrix for the
            provided documents. The rows of this matrix are the terms, and the
            columns are the dates from the original dataframe
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(tokenizer=self.identity_tokenizer, lowercase=False)
        # extract the text_ids
        ids = []
        docs = []
        for doc, id in tokenized_docs_list:
            docs.append(doc)
            ids.append(id)
        # calculate the tf-idf matrix
        tfidf_docs = tfidf.fit_transform(docs)
        tfidf_docs_arr = tfidf_docs.toarray()
        df = pd.DataFrame(tfidf_docs_arr, columns=tfidf.get_feature_names_out())
        df = df.transpose()
        # match up the documents with the original crawl dates
        col_names = []
        for index in ids:
            col_names.append(base_dataframe.at[index, "crawl_date"])
        df.columns = col_names
        return df

    def create_wordcloud_from_terms(self, term_freq, height=640, width=480, title=""):
        """Creates and displays a word cloud from a list of term frequencies.
        Args:
            term_freq: The term frequencies to be used to generate the cloud.
                Designed to work for either a TF-IDF vector or a dictionary of
                raw term frequencies.
        """

        def grey_color_func(
            word, font_size, position, orientation, random_state=None, **kwargs
        ):
            """Recolouring function, sets colours for word cloud words."""
            return "hsl(0, 0%%, %d%%)" % random.randint(1, 45)

        cloud = WordCloud(background_color="white").generate_from_frequencies(term_freq)
        plt.imshow(cloud.recolor(color_func=grey_color_func, random_state=3))
        plt.axis("off")
        fig = plt.gcf()
        fig.set_dpi(100)
        fig.set_size_inches(height / 100, width / 100)
        fig.suptitle(title)
        plt.show()

    def display_wordcloud_options(self):
        """Displays a formfor the creation of word clouds from site content."""
        df = self.data.dropna(subset="content")
        domains = list(df["domain"].value_counts().index)

        domain = widgets.Dropdown(options=domains, description="Domain: ")

        # initial value for the domain set
        domain_set = df[df["domain"] == domains[0]]
        url = widgets.Dropdown(options=domain_set["url"], description="URL:")
        scrapes = domain_set[domain_set["url"] == url.value]
        date = widgets.Dropdown(
            options=scrapes["crawl_date"].dt.date, description="Crawl date:"
        )

        def dropdown_handler_domain(change):
            """Updates the set of URLs available based on change to domain."""
            domain_set = df[df["domain"] == str(change.new)]
            urls = domain_set["url"]
            url.options = urls

        def dropdown_handler_url(change):
            """Updates the set of page captures available based on change to URL."""
            page_set = domain_set[domain_set["url"] == str(change.new)]
            scrapes = page_set["crawl_date"].dt.date
            date.options = scrapes

        domain.observe(dropdown_handler_domain, names="value")
        url.observe(dropdown_handler_url, names="value")

        # preprocessing options
        lemmatize_opt = widgets.Checkbox(value=False, description="Lemmatize words")
        remove_punct_opt = widgets.Checkbox(
            value=False, description="Remove punctation"
        )
        remove_stop_opt = widgets.Checkbox(value=False, description="Remove stopwords")
        preprocess_options_label = widgets.Label("Text preprocessing options: ")

        wordcloud_options_layout = widgets.VBox(
            [
                domain,
                url,
                date,
                preprocess_options_label,
                remove_punct_opt,
                remove_stop_opt,
                lemmatize_opt,
            ]
        )

        # make the figure settings widgets
        label_plot_setting = widgets.Label("Plot settings")
        # height
        plot_height = widgets.IntText(value=640, description="Height (px)")
        # width
        plot_width = widgets.IntText(value=480, description="Width (px)")
        # title
        plot_title = widgets.Text(description="Title")

        plot_settings_box = widgets.VBox([plot_height, plot_width, plot_title])
        plot_set_layout = widgets.VBox([label_plot_setting, plot_settings_box])

        # make the tab display
        tab = widgets.Tab(children=[wordcloud_options_layout, plot_set_layout])
        [
            tab.set_title(i, title)
            for i, title in enumerate(["Wordcloud Settings", "Plot Settings"])
        ]

        def wordcloud_btn_handler(btn):
            """Accepts form input, processes data and creates word cloud."""
            # get settings
            crawl_date = date.value
            crawl_date = crawl_date.strftime("%Y-%m-%d")
            # preprocess
            print("Processing data... (this may take a few minutes)")
            df, token_docs = self.preprocess_content(
                domain_set,
                lemmatize=lemmatize_opt.value,
                remove_punct=remove_punct_opt.value,
                remove_stop=remove_stop_opt.value,
            )

            doc_freqs = self.get_token_freq(token_docs, df)
            # display
            print("Creating Word Cloud ... ")
            self.create_wordcloud_from_terms(
                doc_freqs[crawl_date],
                height=plot_height.value,
                width=plot_width.value,
                title=plot_title.value,
            )

        wordcloud_btn = widgets.Button(description="Create Word Cloud")
        wordcloud_btn.on_click(wordcloud_btn_handler)

        # display the tab + button
        display(tab)
        display(wordcloud_btn)

    def display_top_files(self):
        """Displays most frequently occurring files, by MD5 hash or URL."""

        md5_vals = self.data["md5"].value_counts()
        url_vals = self.data["url"].value_counts()
        slider_lim = max(len(md5_vals), len(url_vals))

        def top_files(col, n):
            """Prints the top n most frequently occurring values from column col."""
            col = col.lower()
            if col == "md5":
                print(
                    f"There are {len(md5_vals)} different files by MD5 hash in the dataset."
                )
                print(
                    f"The top {min(n, len(md5_vals))} files and the number of times they occur in the dataset are shown below."
                )
                print(md5_vals.head(n))
            elif col == "url":
                print(
                    f"There are {len(url_vals)} different files by URL in the dataset."
                )
                print(
                    f"The top {min(n, len(url_vals))} files and the number of times they occur in the dataset are shown below."
                )
                print(url_vals.head(n))
            else:
                print("Unsupported column.")

        type_choice_label = widgets.Label(value="By:")
        type_choice = widgets.Dropdown(options=["MD5", "URL"], value="MD5")
        slider_label = widgets.Label(value="Number of files: ")
        # TODO: see if there is a way to dynamically change the max on the slider
        # based on the dropdown menu selection (since the number may be
        # different for MD5 vs URL)
        n_slider = widgets.IntSlider(value=10, max=slider_lim)
        type_choice_layout = widgets.HBox([type_choice_label, type_choice])
        slider_layout = widgets.HBox([slider_label, n_slider])
        out = widgets.interactive_output(top_files, {"n": n_slider, "col": type_choice})

        # try to dynamically update the slider max based dropdown state
        def dropdown_change(change):
            """Listens for change in dropdown, updates slider max to appropriate value."""
            if change["type"] == "change" and change["name"] == "value":
                if change["new"] == "MD5":
                    n_slider.max = len(md5_vals)
                elif change["new"] == "URL":
                    n_slider.max = len(url_vals)

        type_choice.observe(dropdown_change)

        layout = widgets.VBox([type_choice_layout, slider_layout])
        display(layout)
        display(out)

    def display_file_summary(self):
        """Prints summary info about a file derivative."""
        num_files = len(self.data.index)
        domain_counts = self.data["domain"].value_counts()
        num_domains = len(domain_counts)
        top_n = num_domains if num_domains <= 10 else 10

        extension_counts = self.data["extension"].value_counts()
        num_extensions = len(extension_counts)

        text_output = widgets.Output()
        number_of_files_output = widgets.Output()

        # increase the right margin to add space between the chart and the table
        number_of_files_output.layout.margin = "10px 30px 10px 10px"

        extensions_output = widgets.Output()
        num_files_table_output = widgets.Output()
        num_files_output_panes = widgets.HBox(
            [number_of_files_output, num_files_table_output]
        )
        num_files_lbl = widgets.Label(value="Number of files in the derivative")
        detail_dropdown = widgets.Dropdown(
            description="By:",
            options=["Domain", "Domain and extension"],
            value="Domain",
        )

        num_files_layout = widgets.VBox(
            [num_files_lbl, detail_dropdown, num_files_output_panes]
        )

        tab = widgets.Tab(children=[text_output, num_files_layout, extensions_output])
        [
            tab.set_title(i, title)
            for i, title in enumerate(
                ["Overall Summary", "Number of Files", "File Extensions"]
            )
        ]

        with text_output:
            print(
                f"This derivative contains information about {num_files} files from {num_domains} different domains."
            )
            if num_domains == top_n:
                print("The number of files per domain are as follows: ")
            else:
                print(f" The domains with the most files are: ")
            print(domain_counts.head(top_n))
            print()

            print(
                f"There are {num_extensions} different file extensions represented in this derivative."
                "\nThey are as follows:"
            )
            print(extension_counts)

        # visual summary info
        def on_change(change):
            """Listens for change in plot to be drawn. Changes plot accordingly."""
            if change["type"] == "change" and change["name"] == "value":
                # print("changed to %s" % change['new'])
                if change["new"] == "Domain":
                    number_of_files_output.clear_output(wait=True)
                    # display(detail_dropdown)
                    plt.ioff()
                    with number_of_files_output:
                        self.data["domain"].value_counts().plot(
                            title="Number of files by domain", kind="bar"
                        )
                        plt.show()
                    plt.ion()
                elif change["new"] == "Domain and extension":
                    # by domain and extension
                    number_of_files_output.clear_output(wait=True)
                    # display(detail_dropdown)
                    plt.ioff()
                    with number_of_files_output:
                        df = (
                            self.data.groupby(["domain", "extension"])
                            .size()
                            .unstack(fill_value=0)
                        )
                        # ensure that the data is sorted by the total number of files, in descending order
                        df_sort = (
                            self.data.groupby(["domain"])
                            .count()
                            .sort_values(by="url", ascending=False)
                        )
                        df = df.reindex(index=df_sort.index, level=0)
                        # create the plot
                        df.plot(
                            title="Number of files by domain, extension",
                            kind="bar",
                            stacked=True,
                        )
                        plt.show()
                    plt.ion()

        detail_dropdown.observe(on_change)

        # by domain -- start with just this one
        plt.ioff()
        with number_of_files_output:
            self.data["domain"].value_counts().plot(
                title="Number of files by domain", kind="bar"
            )
            plt.show()
        plt.ion()

        with num_files_table_output:
            display(widgets.Label("Number of files by extension and domain"))
            display(
                self.data.groupby(["domain", "extension"]).size().unstack(fill_value=0)
            )

        # visual display of extensions data
        def autopct(pct):
            """Format labels for charts. Only display labels if greater than 1%."""
            return ("%.1f" % pct + "%") if pct > 1 else ""

        plt.ioff()
        with extensions_output:
            self.data["extension"].value_counts().plot(
                kind="pie",
                autopct=autopct,
                ylabel="",
                title="Files by extension",
            )
            plt.show()
        plt.ion()

        display(tab)

    def same_hash_different_domains(self):
        """Print hashes of files appearing in than one domain, and the domains in which they appear."""
        by_md5 = self.data.groupby("md5")
        printed_header = False
        header = "{: <32s}\t{}".format("MD5 Hash", "Domains")
        for md5, frame in by_md5:
            if len(frame) > 1:
                domains = frame["domain"].unique()
                if len(domains) > 1:
                    if not printed_header:
                        print(header)
                        printed_header = True
                    print(f"{md5}\t{', '.join(domains)}")
        if not printed_header:
            print("No hashes shared between domains were found in this dataset.")

    # image specific functionality
    def build_archive_link_string(self, record, temporal=False):
        """Create a link to the Wayback Machine to view the image in the record.

        Args:
            record: The row of the data frame with data on the record to get the
                link for
            temporal: Boolean, whether to return a link to the Wayback Machine
                for the specific date in the record, or a link to the temporal
                distribution of the image in the record

        Returns:
            archive_url: The string URL for the image
        """
        if temporal:
            time = "*"
        else:
            time = record["crawl_date"].strftime("%Y%m%d%H%M%S")
        url = record["url"]
        base_url = "http://web.archive.org/web/"
        archive_url = base_url + time + "/" + url
        return archive_url

    def get_archive_link_by_hash(self, hash, temporal=False):
        """Gets link to the Wayback Machine for the image specified by its hash.

        Locates the first record in the data with a matching hash and builds the
        link string.

        Args:
            hash: The string hash of the image to get the link for.
            temporal: Boolean indicating whether to get a link to the temporal
                distribution of the image, or simply the image link.

        Returns:
            The link to the image on the Wayback Machine.

        Raises:
            IndexError if a record with a matching URL cannot be found in the
            loaded dataset.
        """
        # get the first matching record for this hash
        try:
            record = self.data.loc[self.data["md5"] == hash].iloc[0]
        except IndexError:  # occurs if unable to find matching data in the df
            raise  # throw the error back to the calling context
        return self.build_archive_link_string(record, temporal)

    def get_archive_link_by_url(self, url, temporal=False):
        """Gets link to the Wayback Machine for the image specified by its URL.

        Locates the first record in the data with a matching URL and builds the
        link string.

        Args:
            url: The string URL of the image to get the link for.
            temporal: Boolean indicating whether to get a link to the temporal
                distribution of the image, or simply the image link.

        Returns:
            The link to the image on the Wayback Machine.

        Raises:
            IndexError if a record with a matching URL cannot be found in the
            loaded dataset.
        """
        try:
            record = self.data.loc[self.data["url"] == url].iloc[0]
        except IndexError:
            raise
        return self.build_archive_link_string(record, temporal)

    def view_image(self):
        """Displays image specified by user input of MD5 hash or URL."""
        # by dropdown
        type_choice = widgets.Dropdown(
            description="By", options=["MD5", "URL"], value="MD5"
        )
        # text field
        record_txt = widgets.Text(
            description="MD5/URL: ",
            placeholder="Enter the URL or MD5 hash of image",
        )
        # button
        display_btn = widgets.Button(description="View Image")
        # output pane for the image, allows it to be cleared between displaying subsequent images.
        out = widgets.Output()

        def display_image(btn):
            """On-click function for the "View Image" button.

            Clears the output pane, creates and displays the image and a
            corresponding link to the Wayback machine as HTML widgets.
            """
            link = ""
            user_input = record_txt.value.strip()
            # ensure the output pane is cleared of any previous output before trying to display new output
            out.clear_output()
            try:
                if type_choice.value == "MD5":
                    link = self.get_archive_link_by_hash(user_input)
                else:
                    link = self.get_archive_link_by_url(user_input)
            except IndexError:
                with out:
                    print(
                        f"Image matching the input {'MD5 hash' if type_choice.value == 'MD5' else 'URL'} unable to be located in the data."
                    )
                    print(
                        "Check input for typos and ensure that the correct type (MD5 or URL) has been selected."
                    )
                return

            with out:
                print(f"The image can be viewed at: {link}")
                image = widgets.HTML(value=f'<img src="{link}">')
                display(image)

        display_btn.on_click(display_image)
        display(type_choice)
        display(record_txt)
        display(display_btn)
        display(out)

    def get_image_temporal_distribution(self):
        """Generates a link to the temporal distribution for user specified image."""
        # by dropdown
        type_choice = widgets.Dropdown(
            description="By", options=["MD5", "URL"], value="MD5"
        )
        # text field
        record_txt = widgets.Text(
            description="MD5/URL: ",
            placeholder="Enter the URL or MD5 hash of image",
        )
        # button
        generate_btn = widgets.Button(description="Get Distribution Link")
        # output pane for the image, allows it to be cleared between displaying
        # subsequent images.
        out = widgets.Output()

        def generate_link(btn):
            """On-click function for the "Get Distribution Link" button.

            Clears the output pane, creates and displays the link to the image
            distribution.
            """
            link = ""
            user_input = record_txt.value.strip()
            # ensure the output pane is cleared of any previous output before
            # trying to display new output
            out.clear_output()
            try:
                if type_choice.value == "MD5":
                    link = self.get_archive_link_by_hash(user_input, temporal=True)
                else:
                    link = self.get_archive_link_by_url(user_input, temporal=True)
            except IndexError:
                with out:
                    print(
                        f"Image matching the input {'MD5 hash' if type_choice.value == 'MD5' else 'URL'} unable to be located in the data."
                    )
                    print(
                        "Check input for typos and ensure that the correct type (MD5 or URL) has been selected."
                    )
                return

            with out:
                print("The temporal distribution for this image can be viewed at: ")
                print(link)

        generate_btn.on_click(generate_link)
        display(type_choice)
        display(record_txt)
        display(generate_btn)
        display(out)

    def display_download_images(self):
        """Displays a form to download top n occurring images from the dataset."""
        label_output = widgets.Label(value="Output folder:")
        text_output = widgets.Text(placeholder="Enter output folder name")
        output_layout = widgets.HBox([label_output, text_output])

        by_choice_label = widgets.Label(value="By: ")
        by_choice = widgets.Dropdown(options=["MD5", "URL"], value="MD5")
        by_choice_layout = widgets.HBox([by_choice_label, by_choice])

        n_md5 = len(self.data["md5"].unique())
        n_url = len(self.data["url"].unique())
        max_n = n_md5  # since the dropdown defaults to MD5, start here by default
        default_n = 10 if 10 < max_n else max_n
        n_slider = widgets.IntSlider(value=default_n, min=1, max=max_n)
        slider_label = widgets.Label(value="Number of images to download: ")
        slider_layout = widgets.HBox([slider_label, n_slider])

        download_btn = widgets.Button(description="Download images")
        out = widgets.Output()

        def dropdown_change(change):
            """Listens for change in dropdown, updates slider max to appropriate value."""
            if change["type"] == "change" and change["name"] == "value":
                if change["new"] == "MD5":
                    n_slider.max = n_md5
                elif change["new"] == "URL":
                    n_slider.max = n_url

        by_choice.observe(dropdown_change)

        def on_click_download(btn):
            """Accepts user input, begins download."""
            by_url = True if by_choice.value == "URL" else False
            out.clear_output()
            output_folder = text_output.value.strip()
            if output_folder == "":
                with out:
                    clear_output()
                    print("Please specify an output folder to save images to.")
                return
            with out:
                clear_output()
                print(
                    f"Downloading top {n_slider.value} images to {path + output_folder}"
                )
                print("This may take some time!")
                self.download_top_n_images(
                    n_slider.value, text_output.value.strip(), by_url=by_url
                )
                print("\nImages downloaded.")

        download_btn.on_click(on_click_download)
        layout = widgets.VBox([output_layout, by_choice_layout, slider_layout])
        display(layout)
        display(download_btn)
        display(out)

    def download_image(self, url, filepath):
        """Downloads the image from url, saves to the file specified by filepath.

        Args:
            url: The string URL to download the image from.
            filepath: The filepath/filename to save the image to.
        """

        response = requests.get(url)
        if not response.ok:
            print(
                f"Request failed: Unable to retrieve image with URL {'http' + url.split('/http', maxsplit = 1)[1]}."
            )
            print(
                f"Request status code {response.status_code}. Reason: {response.reason}"
            )
            if "This URL has been excluded from the Wayback Machine." in response.text:
                print("Image not available from Wayback Machine.")
            return

        img_data = response.content
        with open(filepath, "wb") as file_handle:
            file_handle.write(img_data)

    def download_top_n_images(self, n, out_folder, by_url=True):
        """Downloads the top n images and saves them to the output folder.

        Args:
            n: The number of images to download, the most frequently occurring n
            out_folder: The string representing the folder name to place the
                images in
            by_url: Boolean indicating if the "top" occurrances should be
                determined by URL or MD5 hash

        """
        output_folder_path = Path(path + "/" + out_folder)
        if not output_folder_path.exists():
            output_folder_path.mkdir()

        if by_url:
            field = "url"
        else:
            field = "md5"

        top_values = self.data[field].value_counts().head(n).index
        prog_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=n,
            step=1,
            bar_style="info",
            orientation="horizontal",
        )
        prog_label = widgets.Label(value="Download progress: ")
        prog_label2 = widgets.Label(value=f"0/{n}")
        prog_layout = widgets.HBox([prog_label, prog_bar, prog_label2])
        display(prog_layout)
        count = 0
        for value in top_values:
            record = self.data.loc[self.data[field] == value].iloc[0]
            filename = record["domain"].replace(".", "_") + "_"
            filename += record["filename"]
            if by_url:
                url = self.get_archive_link_by_url(value)
            else:
                url = self.get_archive_link_by_hash(value)
            self.download_image(url, (output_folder_path / filename))
            count += 1
            prog_bar.value = count
            prog_label2.value = f"{count}/{n}"

    ###
    ### Topic Modelling Additions
    ###
    def set_LDA_model_topics(self):
        """Sets the topic model number of topics for Analyzer Object"""
        t_choice = widgets.BoundedIntText(
            value=5,
            min=2,
            max=25,
            step=1,
            # description = "How many topics for LDA Model?",
            disabled=False,
        )
        t_Button = widgets.Button(description="Set")

        def btn_set_topics(btn):
            self.number_LDA_Topics = t_choice.value
            print("Topics Set... Ready to prepare model")

        t_label = widgets.Label("Topics    ")
        t_Button.on_click(btn_set_topics)
        display(widgets.HBox([t_label, t_choice, t_Button]))
