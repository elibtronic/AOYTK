""" AOY-TK module. Provides functions and forms to simplify web-archive analysis. 
"""
# AOY-TK Module
import ipywidgets as widgets 
import requests
import os
import pandas as pd
import matplotlib as plt 


# General purpose functions.
def display_path_select(): 
    """Displays a text box to set the default path for reading / writing data
    """
    txt_path = widgets.Text(description="Folder path:")
    def btn_set_path(btn): 
        global path
        path = txt_path.value
        print(f"Folder path set to: {path}")
    btn_txt_submit = widgets.Button(description="Submit")
    btn_txt_submit.on_click(btn_set_path)
    display(txt_path)
    display(btn_txt_submit)

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
        for f in os.scandir(output_folder): 
            if f.path.split("/")[-1] == "_SUCCESS": 
                # indicate that the derivative was generated successfully
                success = True
                # remove the success indicator file
                os.remove(f.path)
            if f.path.split(".")[-1] == file_type: 
                source_file_name = source_file.split(".")[0]
                source_file_name = source_file_name.split("/")[-1]
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
        data_files = [x for x in os.listdir(path) if x.endswith((".warc", ".arc", "warc.gz", ".arc.gz"))]
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
        
    def load_data(self, filename):
        """Load a datafile to work with. 
        """
        self.data = pd.read_csv()
        