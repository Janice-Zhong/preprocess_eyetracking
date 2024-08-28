# preprocess_eyetracking

Analysis Steps (for each participant)

1.Reading Eye Movement Data:

->Convert .edf files to .asc, or use other toolkits to read the .edf files directly.
->When reading the samples, transform the sample coordinates. Eyelink records coordinates with the screen's top-left as the origin, so the coordinates need to be inverted vertically to set the bottom-left as the origin, making upward the positive axis.

2.Preprocessing Eye Movement Data: For each run:

->Blink: Identify all blink onsets and offsets, and perform linear interpolation on the data within ±0.05 seconds around each blink.
->Extract the eye movement data during the experiment based on the event markers in the eye-tracking record (from trial_1 to trial_432).
->Drift: Apply linear fitting to remove drift.
->Pixel to Degree: Convert gaze coordinates from pixel units to visual degrees.

3.Organizing Eye Movement Data by Position Condition: For each position:

->Find the onset and offset times for the face stimuli appearing at that position across all sessions. The position information for each run is stored in a behavioral results .mat file.
->Extract the eye movement data from all trials that meet the criteria.

4.Analysis: For each position:

->Plot a 2D histogram of eye positions and fit a 2D Gaussian distribution to the histogram. Draw the 95% confidence interval contour.
->Calculate the area of the ellipse within the confidence interval.
