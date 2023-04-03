
# Meeting Notes

## Jan 9

- WAC2023
  - Abstract in Google Drive
  - February 15, sign - ups [via](https://netpreserve.org/ga2023/registration/)
  - March 20 Video is Due
  - May 3 is conference day

- Archives Unleashed Project
  - Draft 2 has been completed, shared with SL

- Semester planning
  - SL to send general availability for this semester to TR

- Next Steps
  - TR to look at website for project, documentation and Topic Modelling (using iPi widgets)
  - SL to debug existing notebook, create analysis notebook

## Jan 16

- Demo of progress so far
  - Creation of derivative
- Discussion of how to encapsulate code?
  - Python classes?
  - 'Flat files'?

- Next Steps
  - SL will continue the derivative extraction workflow
  - TR will continue on web development, first crack at using derivative for analysis


## Jan 23

- Reviewing pySpark _error_, of lack of header, 
- Main module is now object oriented, less **shananigans** required for the generative

- Next Steps
  - TR
   - will see if pySpark thing is actually an error
   - will provide an alternative WARC
   - will look at **Derivative Notebook** to see if changes can be made
  - SL
   - provide pySpark details to TR
   - further development

## Jan 30

 - Temporary Logo completed
 - pdoc to generate 'api' documentation [https://pdoc3.github.io/pdoc](https://pdoc3.github.io/pdoc/doc/pdoc/#gsc.tab=0)
 - Porting to a cleaner Colab experience
 - Work on Analyzer Notebook is underway
 - Decision on default path. Will be set to `content/drive/MyDrive/AOY`. Widgets will be updated to reflect that


 - Next Steps
  - TR
    - continue to port to Colab stuff
  - SL
    - continue to work on Analyzer Notebook
    - Also Workclouds!

## Feb 13

- FJ will make a text analysis notebook. Working on root directory as `/AOY`. Will use same test data set SL has been developing with
- SL will brush up on widget that will be reused. Has made progress on Analyzer notebook, and issues
- TR will look at the CSV output bug and put together prototype of GIT cells
- Development will go on to the idea of 'branches'. FJ, SL, TR, will each create a different branch and work on that 


## Feb 27
 
 - SL, widget work and other things. Changes are pushed to a new branch, header to that csv, bug about dissappearing file
 - TR, slides for IIPC, continue work on Topic Modelling

- Next Steps
  - Break down the slides etc. Who will demo what?
  - Further work on 

## March 6

  - IIPC video is due on March 20
  - SL derivative creation going well, functionality for headers has been added, graph
  - TR did most of a topic modelling book, will get FJ to use the same as well.

  - Next Steps
    - Record demos (hopefully 4 for the presentation, two from SL, 1 from TR & FJ)
    - Slides for next week and demo vids
    - We'll all use the same [5000 sample](https://raw.githubusercontent.com/BrockDSL/ARCH_Data_Explore/main/niagara_sample.csv)

## March 20

  - IIPC Video has been pushed back to 24th
  - 4/5 of the video is completed
  - Fletcher shared good resource: https://github.com/blueprints-for-text-analytics-python/blueprints-text
  - How to standardize pipeline for text processing (will discuss next time what that should look like)
  - How to encapsulate into the object, how much to encapsulate it


  - Next Steps
    - TR will focus on finding journal title that will be a good place for a write-up
    - SL will continue to work on notebooks, working with Colab Pro now instead of regular

## March 27

  - IIPC video has been completed and uploaded
  - [JOSS](https://joss.theoj.org/) as journal venue we'll explore in May
  - SL, 2 design considerations
     - Save-State Paradigm SSP, will write developer doc describing how this works plus demo if possible 
     - Stop word sets, and custom stop words will fold into SSP. Custom stopwords in a text file
  - TR, 
     - identify a WARC/dataset to build documentation on
     - sample derviatves for each of the 4 Collection with exploratory Notebook

## April 3

  - SL, first draft of [design notes](https://github.com/BrockDSL/AOYTK/blob/main/design_notes.md), will continue on code development
  - TR, will inquiry about timelines for MoM

