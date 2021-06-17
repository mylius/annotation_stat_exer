# Statistical Analisys of Annotations Data

## Dependencies

The following packages are required to run the script:
* pandas
* numpy
* sklearn
* plotly

To install the packages run the following:
```
  pip install pandas
  pip install numpy
  pip install sklearn
  pip install plotly
```
## Running the Script

Execute the script by navigating to the folder and running
```
python statistics.py
```
or
```
python statistics.py -p
```
if you want the data to be plotted.

On first execution it will take some time to process the data. After this the data will be pickled and automatically retrieved on subsequent executions.
If you want to reinitialize the data run:
```
python statistics.py -r
```
