This folder contains the scrubbed data from the Original Data folder.
The steps taken to scrub the data are as follows:
1. The data is copied, the header of the original data is
[timestamp, X, Y, Z, (wrongly) calculated angle in rad, measured temperature (to be corrected for overflow),repetitions, ref angle, ref temperature, opsens temp]

2. The data for other angles than -10, -2, -1, 0, 1, 2, 10 deg is deleted, 10 measurement points are kept for each temperature/angle combination (to prevent bias of the resulting model towards a certain combination)

3. Automated scrubbing will take care of:
3.a. Overflow in the measured temperature
3.b. Calculation of the measured angle from the measured XYZ accelerations
3.c. Deletion of the unnecessary columns (X, Y, Z, (wrongly) calculated angle in rad, repetitions,  opsens temp)

Log:
SB2 angle:2 temp:0 a row was copied as only 9 datapoints were available
SB3 angle:2 temp:40 a row was copied as only 9 datapoints were available
