# IoT_deeplearning

## Data without "log"
Saved in milliseconds since 2017.04.03 in a directory with no log. In the case of json, we added it to the millisecond column. For csv and file types, the last column is millisecond. For example, if today is November 10, 2018 at 2:30:36 pm 43, the millisecond column is calculated as follows:
2:30:36 PM 43 => 14: 30: 36.43 => 14 * 60 * 60 * 1000 + 30 * 60 * 1000 + 36 * 1000 + 43. In other words, change the time and minutes to seconds, change the sum of the seconds * 1000 to milliseconds, and add the measured milliseconds (43) to the result.

## Data for "D"
And we divided the log data into the log data and the ceiling log data.
Data in _D in the directory folder is data consisting only of the ceiling data.

## Copy row data
