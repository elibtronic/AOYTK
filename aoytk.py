# AOY-TK Module
import ipywidgets as widgets 
import requests

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

