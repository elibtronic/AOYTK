# Design Notes for AOYTK 

## Save-State Paradigm 
To prevent recomputing results that may be used frequently over the course of analysis, or which may be used for further analysis in other software, these results should be saved to disk. 

The proposed structure is outlined as follows: 

1. For each project created using AOYTK, specify a `project folder`. This would be a reworking of the existing method for setting a working directory. This project folder will be the root folder for a given project. There should be one such folder for each W/ARC file being worked on. 

2. Within the project folder, will be a series of sub folders, corresponding to different types of analysis in the project. The main folder types will be `text`, `collection`, `network` and `file_formats` corresponding to the 4 kinds of derivatives and analysis notebooks supported by the toolkit. 
   - These folders can be created as needed throughout the analysis process.

3. The general proposed structure for creating/accessing resources would be as follows: 
    - When a result is needed for analysis, it should be accessed through an accessor function. This function will need to determine the following things:
      - Does this resource exist in memory already? 
      - If it's not been loaded into memory in this session, can it be found in it's appropriate folder location and loaded in from file for use
      - If neither of the above apply, the accessor should generate the resource, save it to the appropriate location (creating the folder structure as necessary), and retain/return the copy in memory for the session