#%%
import os
import re
import glob
import math
import numpy as np
import matplotlib as plt
from scipy.io import loadmat
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    # Define screen parameters
    screen_height_cm = 50
    screen_height = 1080
    screen_width = 1920
    screen_distance_cm = 160
    length_per_pixel_cm = screen_height_cm / screen_height

    # Define the regex pattern for file names
    run_pattern = re.compile(r'run(1|3|5|7|9)')

    # Path to the folder containing all session subfolders
    sessions_folder = '/Volumes/Janice_Z/eyetracking_data/CN040'
    # Smoothing function
    def smooth_data(data, sampling_rate, window_length_ms=50):
        window_length_samples = int(sampling_rate / 1000 * window_length_ms)
        if window_length_samples > data.shape[0]:
            window_length_samples = data.shape[0]
        smth_kernel = np.ones(window_length_samples) / window_length_samples 
        smoothed_data = np.zeros(data.shape)
        for i in range(data.shape[1]):
            smoothed_data[:, i] = np.convolve(data[:, i], smth_kernel, mode='same')
        return smoothed_data

    # Function to process each session
    def process_session(session_folder, run_pattern):
        preprocessed_data = {}
        deleted_trials = []

        # Get all files in the folder
        all_files = glob.glob(os.path.join(session_folder, '*run[13579]*.asc')) + \
                    glob.glob(os.path.join(session_folder, '*run[13579]*.mat'))

        # Process each file
        for file_path in all_files:
            match = run_pattern.search(file_path)
            if match:
                run = match.group()
                if f'run{run}' not in preprocessed_data:
                    preprocessed_data[f'run{run}'] = {}

                if file_path.endswith('.asc'):
                    trial_data = {}
                    blink_indices = set()
                    with open(file_path, 'r') as file:
                        current_trial = None
                        line_index = 0
                        for line in file:
                            if 'trial_' in line:
                                current_trial = line.strip()
                                trial_data[current_trial] = []
                                line_index = 0
                            elif current_trial:
                                elements = line.strip().split()
                                # delete blink data
                                if len(elements) >= 5 and all(element.replace('.', '', 1).isdigit() or element == '.' for element in elements[:5]):
                                    numeric_elements = [float(element) if element != '.' else None for element in elements[:5]]
                                    if numeric_elements[1] == 0 and numeric_elements[2] == 0:
                                        blink_indices.update(range(max(0, line_index - 50), line_index + 51))  
                                    if line_index not in blink_indices:  
                                        trial_data[current_trial].append(numeric_elements)
                                    line_index += 1
                                    if (line_index > 250) & ('432' in current_trial):
                                        break

                    run_data_x = np.array([]) 
                    run_data_y = np.array([])
                    trial_ind = np.array([]) 
                    for trial, samples in trial_data.items():
                        valid_samples = [sample for sample in samples if sample[1] is not None and sample[2] is not None]
                        if len(valid_samples) > 0:
                            run_data_x = np.append(run_data_x, np.array(valid_samples)[:,1])
                            run_data_y = np.append(run_data_y, np.array(valid_samples)[:,2])
                            trial_ind = np.append(trial_ind, np.ones(len(valid_samples))*(int(re.search(r'\d+$', trial).group())-1))
                        else:
                            #print(f"No valid data for trial {trial}, skipping.")
                            deleted_trials.append(trial)
                    if len(run_data_x) == 0:
                        continue
                    run_data_raw = np.vstack([run_data_x,run_data_y]).T
                    preprocessed_data[f'run{run}']['run_data_raw'] = run_data_raw
                    preprocessed_data[f'run{run}']['trial_ind'] = trial_ind.astype(int)

                    #  transform pixel to degree
                    unit = math.atan(1 * length_per_pixel_cm / screen_distance_cm) * 180 / np.pi
                    # coordinate centering
                    degree_x = (run_data_raw[:,0] - screen_width / 2) * unit
                    degree_y = (screen_height - run_data_raw[:,1] - screen_height / 2) * unit
                    degree_data=np.vstack([degree_x, degree_y]).T
                    preprocessed_data[f'run{run}']['degree_data'] = degree_data

                    # delete drift
                    slope_x, intercept_x, _, _, _ = linregress(range(degree_data.shape[0]), list(degree_data[:,0]))
                    slope_y, intercept_y, _, _, _ = linregress(range(degree_data.shape[0]), list(degree_data[:,1]))

                    corrected_x = degree_data[:,0] - (slope_x * np.arange(0,degree_data.shape[0]) + intercept_x)
                    corrected_y = degree_data[:,1] - (slope_y * np.arange(0,degree_data.shape[0]) + intercept_y)
                    drift_data = np.vstack([corrected_x, corrected_y]).T
                    preprocessed_data[f'run{run}']['drift_data'] = drift_data

                    # centered_data
                    detrend_centered_data = drift_data - np.array([np.median(drift_data[:,0]),np.median(drift_data[:,1])])
                    preprocessed_data[f'run{run}']['detrend_centered_data'] = detrend_centered_data        

                    # smmoth
                    sampling_rate = 500
                    smoothing_data = smooth_data(detrend_centered_data, sampling_rate)
                    preprocessed_data[f'run{run}']['smoothing_data'] = smoothing_data

                # align with flag data
                elif file_path.endswith('.mat'):
                    all_samples_with_flags = []
                    mat_data = loadmat(file_path)
                    if 'this_posi_list' in mat_data:
                        matrix = mat_data['this_posi_list']
                        flattened_matrix = matrix.flatten()
                    for run_key, run_data in preprocessed_data.items():
                        if 'smoothing_data' in run_data:
                            smoothing_data = run_data['smoothing_data']
                            trial_ind = run_data['trial_ind']
                            trial_flag = flattened_matrix[trial_ind]
                            for sample in zip(smoothing_data[:, 0], smoothing_data[:, 1], trial_flag):
                                trial_num = re.search(r'\d+$', str(sample[-1]))
                                if trial_num and trial_num.group() not in deleted_trials:
                                    all_samples_with_flags.append(sample)
        return all_samples_with_flags

    # combining all data
    all_sessions_data = []
    sessions_folders = [os.path.join(sessions_folder, d) for d in os.listdir(sessions_folder) if os.path.isdir(os.path.join(sessions_folder, d))]

    for session_folder in sessions_folders:
        all_sessions_data.extend(process_session(session_folder, run_pattern))

    classified_data = {}
    for sample in all_sessions_data:
        flag = sample[-1]
        if flag not in classified_data:
            classified_data[flag] = []
        classified_data[flag].append(sample)

    # 2d_faussian_fitting 
    def gaussian_2d_with_rotation(xy, A, x0, y0, sigma_x, sigma_y, theta):
        x, y = xy
        x_rot = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
        y_rot = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
        g = A * np.exp(-(x_rot)**2 / (2 * sigma_x**2) - (y_rot)**2 / (2 * sigma_y**2))
        return g.flatten()  

    def plot_heatmap_with_contour(ax, x, y, title=''):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=100, range=[[-6, 6], [-6, 6]], density=True)

        x_center = 0.5 * (xedges[1:] + xedges[:-1])
        y_center = 0.5 * (yedges[1:] + yedges[:-1])

        xg, yg = np.meshgrid(x_center, y_center)

        # fitting with constraints
        try:
            popt, pcov = curve_fit(
                gaussian_2d_with_rotation, (xg, yg), heatmap.flatten(),
                p0=[1, np.mean(x), np.mean(y), np.std(x), np.std(y), 0],
                bounds=(
                    [0, -6, -6, 0, 0, -np.pi/2],  
                    [np.inf, 6, 6, 6, 6, np.pi/2]  
                )
            )

            A, x0, y0, sigma_x, sigma_y, theta = popt

            gaussian_values = gaussian_2d_with_rotation((xg, yg), A, x0, y0, sigma_x, sigma_y, theta)
            gaussian_values = gaussian_values.reshape(100, 100)

            # plot
            ax.grid(True, which='both', color='gray', linestyle='--', linewidth=1)
            ax.set_xticks(np.arange(-6, 6, step=1))
            ax.set_yticks(np.arange(-6, 6, step=1))

            # plot heatmap
            ax.imshow(heatmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='hot')

            threshold = np.max(gaussian_values) * 0.05
            contour = ax.contour(xg, yg, gaussian_values, levels=[threshold], colors='lightsteelblue', linewidths=2)

            # plot 95% contour
            if len(contour.collections) > 0:
                contour_points = contour.collections[0].get_paths()[0].vertices
                major_axis = 2 * np.sqrt(2 * np.log(2)) * sigma_x
                minor_axis = 2 * np.sqrt(2 * np.log(2)) * sigma_y
                area = np.pi * np.abs(major_axis) * np.abs(minor_axis)
                ax.set_title(title + ' - Area: {:.2f}'.format(area))
            else:
                ax.set_title(title + ' - No contours found')
            
            return area

        except (RuntimeError, ValueError) as e:
            ax.set_title(title + ' - Fit Error')
            print(f"Fit error for {title}: {e}")
            return None

        #ax.set_xticklabels([])
        #ax.set_yticklabels([])

     

    for session_folder in sessions_folders:
        try:
            # Get the session number from the folder name
            session_number = re.search(r'session(\d+)', session_folder).group(1)
            # Process the session
            session_data = process_session(session_folder, run_pattern)

            # Generate plot for the session
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            areas = {}
            for flag, ax in enumerate(axes.ravel(), 1):
                data = np.array([sample for sample in session_data if sample[-1] == flag])
                data = data[(np.abs(data[:, 0]) <= 6) & (np.abs(data[:, 1]) <= 6)]
                if data.size > 0:
                    x, y = data[:, 0], data[:, 1]
                    area = plot_heatmap_with_contour(ax, x, y, title=f'Session {session_number}, Flag {flag}')
                    areas[flag] = area 
                    #plot_heatmap_with_contour(ax, x, y, title=f'Session {session_number}, Flag {flag}')

            plt.tight_layout()
            save_folder = '/Users/janicezhong/eyetracking/output'
            save_filename = f'CN040_face_session_{session_number}.png'
            plt.savefig(os.path.join(save_folder, save_filename))
            plt.close()  # Close the current plot to release memory

            # Print the areas for each flag in this session
            print(f"Session {session_number} areas:")
            for flag, area in areas.items():
                print(f'{area:.2f}')
        except Exception as e:
            print(f"Error processing session {session_folder}: {str(e)}")

except Exception as e:
    print(f"Error: {str(e)}")

# %%
