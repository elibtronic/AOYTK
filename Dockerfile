FROM jupyter/pyspark-notebook:spark-3.3.1

# copy in the sample resources
RUN git clone https://github.com/archivesunleashed/aut-resources.git

# download the Archives Unleashed Toolkit
USER root
RUN mkdir -p ${HOME}/data \ 
    && chown ${NB_USER} ${HOME}/data
RUN mkdir -p ${HOME}/aut \ 
    && cd ${HOME}/aut \
    && wget -q "https://github.com/archivesunleashed/aut/releases/download/aut-1.1.1/aut-1.1.1-fatjar.jar" \
    && wget -q "https://github.com/archivesunleashed/aut/releases/download/aut-1.1.1/aut-1.1.1.zip" \
    && chown ${NB_USER} ${HOME}/aut

USER ${NB_USER}

RUN pip install findspark

ENV PYSPARK_PYTHON=/opt/conda/bin/python
ENV PYSPARK_DRIVER_PYTHON=/opt/conda/bin/python
ENV PYSPARK_SUBMIT_ARGS = --jars ${HOME}/aut/aut-1.1.1-fatjar.jar --py-files ${HOME}/aut/aut-1.1.1.zip pyspark-shell