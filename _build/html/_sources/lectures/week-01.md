# W1 problem

This week our ultimate goal is to imlpement a data-cleaning/validating loop.   We will have a **list** of data, that ostensibly represents reported heights in different formats/units.  We want to convert each report into one representation: height as an integer number of millimeters.  We will **loop** through the list, and for each item we will check if it is a valid height, and identify how that height was reported.  We will then convert each reported height into a common format, and output a list containing the cleaned data.   This kind of process is essential to any kind of data processing computations you might undertake.

Specifically, heights in this list may be encoded as follows :   

- decimal numbers between 2.75 and 9 indicate height in feet.
- decimal numbers between 0.83 and 2.75 indicate height in meters.
- decimal numbers between 12 and 108 indicate height in inches.
- integers between 83 and 275 indicate height in centimeters
- integers between 830 and 2750 indicate height in milimeters

Any other format is invalid and should be converted to None

This is a horrible coding scheme, never write a data file this way! Although the pretend data format here is pretty horrid, monstrosities like this occur in the [real world](https://twitter.com/minebocek/status/914166872282664960).

## Pseudo-algorithm

We will write out an algorithm for what we want to do, in english:

I. for each item in the raw list  
  1. convert it into milimeters if possible  
  2. add the converted number to the output list

This is a nice start, but we need to be more clear: what is our process for converting?  We know there are 12 inches in a foot, 25.4 millimeters per inch, 10 millimeters per centimeter, 1000 millimeters per meter (if we did not know this, we could look it up quite easily), but the algorithm is not specified.

I. for each item in the raw list  
  1. convert it into milimeters if possible  
     - if item is a decimal between 2.75 and 9 (feet): converted = item * 12 * 25.4
     - if item is a decimal between 0.83 and 2.75 (meters): converted = item * 1000
     - if item is a decimal between 12 and 108 (inches): converted = item * 25.4
     - if item is an integer between 283 and 275 (cm): converted = item * 10
     - if item is an integer between 830 and 2750 (mm): converted = item
     - otherwise: converted=None 
  2. add the converted number (as an integer) to the output list

Now we have a fairly clear algorithm written out.  What we are missing is any knowledge of Python itself: we don't know what kinds of words Python understands, or how it represents data of the different sorts.  We need to learn how to speak Python, and in particular:   
- what is a list as far as Python is concerned?  
- how do we do something to each item of a list?  
- how can we tell Python to check if an item is a decimal or an integer?  
- how can we ask Python if an item is in some numerical range?  
- how can we assign a new value to converted?  
- how can we tell Python to represent that value as an integer?  
- how can we make a new list and add items to it?

We will learn enough Python this week to be able to convert the algorithm we wrote in English, above, into Python. 