""" AOY-TK module. Provides functions and forms to simplify web-archive analysis. 
"""
# AOY-TK Module
import ipywidgets as widgets 
import requests
import os
import pandas as pd
import matplotlib as plt 
import re
from google.colab import drive
from IPython.display import clear_output

# Global path variable -- a default for Google Drive usage
path = "/content/drive/MyDrive/AOY/" # default path, can be overwritten by the path-setter widget

# General purpose functions.
def display_path_select(): 
    """Displays a text box to set the default path for reading / writing data
    """
    txt_path = widgets.Text(description="Folder path:", placeholder = "Enter your folder path", value = "/content/drive/MyDrive/AOY/")
    def btn_set_path(btn): 
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

def get_files(main_directory, file_types):
  """Recursively list files of given types from directory + its subdirectories.

    Args: 
      main_directory (str): the root directory to look for files in 
        (including its subdirectories). 
      file_types (tuple of str): file types to match on ex. (".csv", ".parquet", ".pqt")
    
    Returns: 
      a list of the matching file names, including their path relative to the top level directory
      Ex. a file in the top-level directory will only return the file name, 
          a file in a sub directory will include the subdirectory name and 
          the file name. ex. "subdir/a.csv"
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
def download_file(url, filepath='', filename=None, loud=True):
  """Displays a text box to specify URL to download file from.

  Args: 
    url : the URL path to download the file from 
    filepath : the file path specifying the folder to save the file into
    filename : the filename to give to the downloaded file 
               (if None, the filename will be extracted from the URL)
    loud : boolean indicating whether or not to display download progress
  """
  if not filename:
    filename = url.split('/')[-1]
    if "?" in filename: 
        filename = filename.split("?")[0]
  
  r = requests.get(url, stream=True)
  if loud:
    total_bytes_dl = 0
    content_len = int(r.headers['Content-Length'])
    prog_bar = widgets.IntProgress(value=1, min=0, max=100, step=1, bar_style='info',orientation='horizontal')
    print(f'Download progress of {filename}:')
    display(prog_bar)

  with open(filepath + filename, 'wb') as fd:
      for chunk in r.iter_content(chunk_size=4096):
          fd.write(chunk)
          if loud:
            total_bytes_dl += 4096
            percent = int((total_bytes_dl / content_len) * 100.0)
            prog_bar.value = percent
  r.close()
  print("File download completed.")


def display_download_file(): 
    """Display textbox to download file from specified URL.
    """
    txt_url = widgets.Text(description="W/ARC URL: ")
    btn_download = widgets.Button(description = "Download W/ARC")
    def btn_download_action(btn): 
        url = txt_url.value
        if url != '': 
            download_file(url, path + "/") # download the file to the specified folder set in the above section
        else: 
            print("Please specify a URL in the textbox above.")
    btn_download.on_click(btn_download_action)
    display(txt_url)
    display(btn_download)



class DerivativeGenerator: 
    """Creates derivative files from W/ARCs. 
    
    This class contains all of the functions relating to derivative generation."""
    def __init__(self):
        """ Initialize the dependencies for creating derivatives.
        """
        # initialize the PySpark context
        import findspark
        findspark.init()
        import pyspark
        self.sc = pyspark.SparkContext()
        from pyspark.sql import SQLContext
        self.sqlContext = SQLContext(self.sc)

    def create_csv_with_header(self, headers, datafile, outputfile): 
      """ Create a version of datafile with the specified headers. 

      Args: 
        headers: a list of column headers in the desired order
        datafile: the path of the CSV file, without headers, to add headers to
        outputfile: the path of the desired output file 
      """
      import csv
      with open(outputfile, "w", newline = "") as csvfile: 
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(headers)
        with open(datafile, "r") as datafile: 
          reader = csv.reader(datafile)
          for row in reader: 
            writer.writerow(row)

    # a messy first guess at derivative generation
    def generate_derivative(self, source_file, output_folder, file_type="csv", text_filters=0):
        """Create a text derivative file from the specified source file.

        Create a text derivative from the specified W/ARC source file, using the output settings specified. 
        Args: 
            source_file: the path to the W/ARC file to generatet the derivative from 
            output_folder: the name for the output folder to save the derivative into 
                        (Note: this is currently a relative path, the folder will be created as a 
                                sub-folder of the working folder)
            file_type: the file format to save the produced derivative in. 
                    Can be either "csv" or "parquet" 
            text_filters: an integer representing which type of text filtering to apply to the generated derivative. 
                        0 : return the complete text content of each webpage (with HTML tags removed)
                        1 : return the complete text with HTTP headers removed 
                        2 : return the text with the boilerplate removed (boilerplate includes nav bars etc) 
        """ 
        # import the AUT (needs to be done after the PySpark set-up)
        from aut import WebArchive, remove_html, remove_http_header, extract_boilerplate
        from pyspark.sql.functions import col, desc

        # create our WebArchive object from the W/ARC file
        archive = WebArchive(self.sc, self.sqlContext, source_file)

        if text_filters == 0: 
            content = remove_html("content")
        elif text_filters == 1: 
            content = remove_html(remove_http_header("content"))
        else: 
            content = extract_boilerplate(remove_http_header("content")).alias("content")

        archive.webpages() \
            .select("crawl_date", "domain", "url", content) \
            .write \
            .option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ") \
            .format(file_type) \
            .option("escape", "\"") \
            .option("encoding", "utf-8") \
            .save(output_folder)

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
                  if text_filters >= 0 and text_filters <= 2: 
                    headers = ["crawl_date", "domain", "url", "content"]
                  output_path = output_folder + source_file_name + ".csv"
                  self.create_csv_with_header(headers, f.path, output_path)
                  os.remove(f.path)
                else:   
                  os.rename(f.path, output_folder + source_file_name + "." + file_type)
        return success

    def display_derivative_creation_options(self): 
        """ Displays a form to set options for derivative file creation. 

        Displays 4 form elements to select: 
        - any W/ARC file from within the defined working folder to create a derivative of
        - desired type of derivative (i.e. what content to include in the derivative)
        - the output folder for the derivative (will be created within the working directory)
        - the desired output file type (csv or parquet)

        Also displays a button which, on-click, will run generate_derivative(), 
        passing in the settings specified in the form. 
        """
        # file picker for W/ARC files in the specified folder
        data_files = get_files(path, (".warc", ".arc", "warc.gz", ".arc.gz"))
        file_options = widgets.Dropdown(description="W/ARC file:", options =  data_files)
        out_text = widgets.Text(description="Output folder:", value="output/")
        format_choice = widgets.Dropdown(description="File type:",options=["csv", "parquet"], value="csv")
        # text content choices 
        content_options = ["All text content", "Text content without HTTP headers", "Text content without boilerplate"]
        content_choice = widgets.Dropdown(description="Content:", options=content_options)
        content_val = content_options.index(content_choice.value)
        button = widgets.Button(description="Create derivative")

        # this function is defined here in order to keep the other form elements 
        # in-scope and therefore allow for the reading of their values
        def btn_create_deriv(btn): 
            """On-click function for the create derivative button. 

            Retrieves the values from the other inputs on the form and passes them to 
            generate_derivative() to create a derivative file using the selected settings. 
            """
            content_options = ["All text content", "Text content without HTTP headers", "Text content without boilerplate"]
            input_file = path + "/" + file_options.value
            output_location = path + "/" + out_text.value
            content_val = content_options.index(content_choice.value)
            print("Creating derivative file... (this may take several minutes)")
            if self.generate_derivative(input_file, output_location, format_choice.value, content_val):
                print("Derivative generated, saved to: " + output_location)
            else: 
                print("An error occurred while processing the W/ARC. Derivative file may not have been generated successfully.")
        button.on_click(btn_create_deriv)
        display(file_options)
        display(out_text)
        display(format_choice)
        display(content_choice)
        display(button)


class Analyzer: 
    """ Tools for analyzing W/ARC derivatives.
    """
    def __init__(self): 
        # initialize the data attribute to None -- should possibly be an empty dataframe? consult with appropriate design patterns
        self.data = None
        self.number_LDA_Topics = None

    def set_data(self, datafile): 
      """ Sets the data attribute for the Analyzer. 

      Parses columns to appropriate types if applicable. 

      Args: 
        datafile (str): the path to the datafile to analyze. 
      """
      self.data = pd.read_csv(datafile)
      
      # if the crawl_date column is included on the frame, make it a date
      if "crawl_date" in list(self.data): 
        self.data['crawl_date']= pd.to_datetime(self.data['crawl_date']) #, format='%Y%m%d%H%M%S'

    def load_data(self):
        """Load a datafile to work with. 
        """
        # display the options available in the working directory
        # Parquet files are not currently supported, if/when they are, add '".parquet", ".pqt"' to the file ending options 
        label = widgets.Label("Derivative")
        file_options = widgets.Dropdown(description = "", options = get_files(path, (".csv")))
        button = widgets.Button(description = "Select file")
        
        def btn_select_file(btn): 
            selected_file = path + "/" + file_options.value
            print("Loading data...")
            self.set_data(selected_file)
            print(f"Data loaded from: {selected_file}")
        
        button.on_click(btn_select_file)
        display(widgets.HBox([label, file_options, button]))        
    def date_range_select(self):
      """ Display a date range selector for valid dates in the data.
      """
      from IPython.display import display, Javascript
      valid_range = self.data.reset_index()['crawl_date'].agg(['min', 'max'])
      start_label = widgets.Label("Select a start date ")
      start_picker = widgets.DatePicker(description = "", 
                                        value = valid_range["min"], 
                                        disabled = False)
      start_picker.add_class("start-date")
      end_label = widgets.Label("Select an end date ")
      end_picker = widgets.DatePicker(description = "", 
                                      value = valid_range["max"], 
                                      disabled = False)
      end_picker.add_class("end-date")

      js = f"""const query = '.start-date > input:first-of-type';
           document.querySelector(query).setAttribute('min', '{valid_range['min'].strftime('%Y-%m-%d')}');
           document.querySelector(query).setAttribute('max', '{valid_range['max'].strftime('%Y-%m-%d')}'); 
           const q = '.end-date > input:first-of-type';
           document.querySelector(q).setAttribute('min', '{valid_range['min'].strftime('%Y-%m-%d')}');
           document.querySelector(q).setAttribute('max', '{valid_range['max'].strftime('%Y-%m-%d')}');"""  
      script = Javascript(js)

      display(widgets.HBox([start_label, start_picker]))
      display(widgets.HBox([end_label, end_picker]))
      display(script)

    def display_top_domains(self): 
        """Display the most frequently crawled domains in the dataset.
        """
        domain_values = self.data["domain"].value_counts();
        n_domains = len(domain_values)
        def top_domains(n): 
            print(domain_values.head(n))
        n_slider = widgets.IntSlider(value = 10, max = n_domains)
        out = widgets.interactive_output(top_domains, {'n':n_slider})
        print(f"There are {n_domains} different domains in the dataset. ")
        display(n_slider)
        display(out)

###
### Topic Modelling Additions
###
    def set_LDA_model_topics(self):
      """ Sets the topic model number of topics for Analyzer Object
      """
      t_choice = widgets.BoundedIntText(
        value = 5,
        min = 2,
        max = 25,
        step = 1,
        #description = "How many topics for LDA Model?",
        disabled = False,
      )
      t_Button = widgets.Button(description = "Set")
      

      def btn_set_topics(btn): 
          self.number_LDA_Topics = t_choice.value
          print("Topics Set... Ready to prepare model")
      t_label = widgets.Label("Topics    ")
      t_Button.on_click(btn_set_topics)
      display(widgets.HBox([t_label, t_choice,t_Button]))


