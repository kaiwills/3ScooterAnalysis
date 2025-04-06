import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from scipy.stats import norm
from PIL import Image


def calc_angles(angles_in):
    # Takes joint angles -> outputs angles relative to x-axis
    # angles_in = [45, 45, 110, 30, 135]

    a = angles_in[0]
    b = 180 - (angles_in[1] - a)
    c = angles_in[2]
    d = 180 + c + angles_in[3]

    angles_out = [180, a, b, c, d, 0]

    return angles_out


def calc_joint_positions(df, gender, percentile, angles_in):
    
    foot_length = df['FootLength'][f'{gender}{percentile}']  
    lower_leg = df['LowerLeg'][f'{gender}{percentile}']
    thigh_length = df['ThighLength'][f'{gender}{percentile}']
    torso_height = df['TorsoHeight'][f'{gender}{percentile}'] 
    upper_arm = df['UpperArm'][f'{gender}{percentile}'] 
    forearm = df['Forearm'][f'{gender}{percentile}']

    dimensions = [foot_length, lower_leg, thigh_length, torso_height, upper_arm, forearm]
    
    angles = calc_angles(angles_in)

    joints = [(0, 0)]  # starting point for the foot
    
    # Iterate through each body part to calculate the next joint position
    for i in range(len(dimensions)):
        # Extract the current joint position, length of the current body part, and angle
        x_prev, y_prev = joints[-1]
        length = dimensions[i]
        angle = angles[i]
        
        # Calculate the new joint position using trigonometry
        x_new = x_prev + length * np.cos(np.radians(angle))
        y_new = y_prev + length * np.sin(np.radians(angle))
        
        # Add the new joint position to the list of joints
        joints.append((x_new, y_new))

    return joints


def plot_skeleton(joints, dists = True, gender = 'Man', percentile = '50', save=False, overlay=False):
    
    # Plot the skeleton using the joints
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    
    # Plot the body parts as lines between joints
    for i in range(1, len(joints)):
        x_values = [joints[i-1][0], joints[i][0]]
        y_values = [joints[i-1][1], joints[i][1]]
        ax.plot(x_values, y_values, color='red', linewidth = 2)  # Plot lines between joints
    
    # Plot dots at each joint
    joint_x, joint_y = zip(*joints)  # Unzip the x and y coordinates of the joints
    ax.scatter(joint_x, joint_y, color='black')  # Plot the joints as dots
    
    if dists:
        
        if overlay:
            # Horizontal A
            ax.annotate('', 
                xy=(joint_x[3], joint_y[4] + 500),  # Start of the line (xy)
                xytext=(joint_x[6], joint_y[4] + 500),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='<-'))  # Arrow at the start
            ax.annotate('', 
                xy=(joint_x[3], joint_y[4] + 500),  # Start of the line (xy)
                xytext=(joint_x[6], joint_y[4] + 500),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='->'))  # Arrow at the end
            ax.vlines(joint_x[3], joint_y[3], joint_y[4]+500 , linestyle='--', colors='black', linewidth=0.5)
            ax.vlines(joint_x[6], joint_y[6], joint_y[4]+500 , linestyle='--', colors='black', linewidth=0.5)
            ax.text((joint_x[3] + joint_x[6]) / 2, joint_y[4] + 530, "Distance A", ha='center', va='bottom', fontsize=10)

            # Vertical B
            ax.annotate('', 
                xy=( max(joint_x) + 250, joint_y[0]),  # Start of the line (xy)
                xytext=( max(joint_x) + 250, joint_y[6]),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='<-'))  # Arrow at the start
            ax.annotate('', 
                xy=( max(joint_x) + 250, joint_y[0]),  # Start of the line (xy)
                xytext=( max(joint_x) + 250, joint_y[6]),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='->'))  # Arrow at the end
            ax.hlines(joint_y[0], joint_x[0], max(joint_x)+250 , linestyle='--', colors='black', linewidth=0.5)
            ax.hlines(joint_y[6], joint_x[6], max(joint_x)+250 , linestyle='--', colors='black', linewidth=0.5)
            ax.text(max(joint_x) + 280, (joint_y[0] + joint_y[6]) / 2, "Distance B", ha='left', va='center', fontsize=10, rotation=90)

            # Vertical C
            ax.annotate('', 
                xy=(joint_x[3] - 250, joint_y[0]),  # Start of the line (xy)
                xytext=(joint_x[3] - 250, joint_y[3]),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='<-'))  # Arrow at the start
            ax.annotate('', 
                xy=(joint_x[3] - 250, joint_y[0]),  # Start of the line (xy)
                xytext=(joint_x[3] - 250, joint_y[3]),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='->'))  # Arrow at the end
            ax.hlines(joint_y[0], joint_x[0], joint_x[3]-250 , linestyle='--', colors='black', linewidth=0.5)
            ax.hlines(joint_y[3], joint_x[3], joint_x[3]-250 , linestyle='--', colors='black', linewidth=0.5)
            ax.text(joint_x[3] - 340, (joint_y[0] + joint_y[3]) / 2, "Distance C", ha='left', va='center', fontsize=10, rotation=90)

            # Horizontal D
            ax.annotate('', 
                xy=(joint_x[3], min(joint_y) - 200),  # Start of the line (xy)
                xytext=(joint_x[0], min(joint_y) - 200),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='<-'))  # Arrow at the start
            ax.annotate('', 
                xy=(joint_x[3], min(joint_y) - 200),  # Start of the line (xy)
                xytext=(joint_x[0], min(joint_y) - 200),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='->'))  # Arrow at the end
            ax.vlines(joint_x[3], joint_y[3], min(joint_y) - 200 , linestyle='--', colors='black', linewidth=0.5)
            ax.vlines(joint_x[0], joint_y[0], min(joint_y) - 200 , linestyle='--', colors='black', linewidth=0.5)
            ax.text((joint_x[3] + joint_x[0]) / 2, min(joint_y) - 300, "Distance D", ha='center', va='bottom', fontsize=10)

            # Adjust the plot limits dynamically to fit all joints
            x_min = min(joint_x) - 300
            x_max = max(joint_x) + 500
            y_min = min(joint_y) - 100
            y_max = max(joint_y) + 750
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)   

        else:
            # Horizontal A
            ax.annotate('', 
                xy=(joint_x[3], joint_y[4] + 200),  # Start of the line (xy)
                xytext=(joint_x[6], joint_y[4] + 200),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='<-'))  # Arrow at the start
            ax.annotate('', 
                xy=(joint_x[3], joint_y[4] + 200),  # Start of the line (xy)
                xytext=(joint_x[6], joint_y[4] + 200),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='->'))  # Arrow at the end
            ax.vlines(joint_x[3], joint_y[3], joint_y[4]+200 , linestyle='--', colors='black', linewidth=0.5)
            ax.vlines(joint_x[6], joint_y[6], joint_y[4]+200 , linestyle='--', colors='black', linewidth=0.5)
            ax.text((joint_x[3] + joint_x[6]) / 2, joint_y[4] + 230, "Distance A", ha='center', va='bottom', fontsize=10)

            # Vertical B
            ax.annotate('', 
                xy=(max(joint_x) + 250, joint_y[0]),  # Start of the line (xy)
                xytext=(max(joint_x) + 250, joint_y[6]),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='<-'))  # Arrow at the start
            ax.annotate('', 
                xy=(max(joint_x) + 250, joint_y[0]),  # Start of the line (xy)
                xytext=(max(joint_x) + 250, joint_y[6]),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='->'))  # Arrow at the end
            ax.hlines(joint_y[0], joint_x[0], max(joint_x)+250 , linestyle='--', colors='black', linewidth=0.5)
            ax.hlines(joint_y[6], joint_x[6], max(joint_x)+250 , linestyle='--', colors='black', linewidth=0.5)
            ax.text(max(joint_x) + 280, (joint_y[0] + joint_y[6]) / 2, "Distance B", ha='left', va='center', fontsize=10, rotation=90)

            # Vertical C
            ax.annotate('', 
                xy=(joint_x[3] - 250, joint_y[0]),  # Start of the line (xy)
                xytext=(joint_x[3] - 250, joint_y[3]),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='<-'))  # Arrow at the start
            ax.annotate('', 
                xy=(joint_x[3] - 250, joint_y[0]),  # Start of the line (xy)
                xytext=(joint_x[3] - 250, joint_y[3]),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='->'))  # Arrow at the end
            ax.hlines(joint_y[0], joint_x[0], joint_x[3]-250 , linestyle='--', colors='black', linewidth=0.5)
            ax.hlines(joint_y[3], joint_x[3], joint_x[3]-250 , linestyle='--', colors='black', linewidth=0.5)
            ax.text(joint_x[3] - 340, (joint_y[0] + joint_y[3]) / 2, "Distance C", ha='left', va='center', fontsize=10, rotation=90)

            # Horizontal D
            ax.annotate('', 
                xy=(joint_x[3], min(joint_y) - 200),  # Start of the line (xy)
                xytext=(joint_x[0], min(joint_y) - 200),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='<-'))  # Arrow at the start
            ax.annotate('', 
                xy=(joint_x[3], min(joint_y) - 200),  # Start of the line (xy)
                xytext=(joint_x[0], min(joint_y) - 200),  # End of the line (xytext)
                arrowprops=dict(facecolor='black', arrowstyle='->'))  # Arrow at the end
            ax.vlines(joint_x[3], joint_y[3], min(joint_y) - 200 , linestyle='--', colors='black', linewidth=0.5)
            ax.vlines(joint_x[0], joint_y[0], min(joint_y) - 200 , linestyle='--', colors='black', linewidth=0.5)
            ax.text((joint_x[3] + joint_x[0]) / 2, min(joint_y) - 300, "Distance D", ha='center', va='bottom', fontsize=10)

            # Adjust the plot limits dynamically to fit all joints
            x_min = min(joint_x) - 500
            x_max = max(joint_x) + 500
            y_min = min(joint_y) - 400
            y_max = max(joint_y) + 400
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    image = Image.open('Man_Seated.jpeg').convert('RGBA')

    img_width, img_height = image.size
    ratio = img_width / img_height

    desired_width = joint_x[0] - joint_x[4] + 250
    desired_height = desired_width / ratio - 650

    xmin, xmax = joint_x[4] - 200, joint_x[4] + desired_width - 200
    ymin, ymax = joint_y[0] - 100, joint_y[0] + desired_height - 100

    if overlay:
        ax.imshow(image, extent = [xmin, xmax, ymin, ymax], aspect='auto')
    
    plt.title(f"2D Human Skeleton, {gender}, {percentile}%")
    
    if save:
        plt.savefig(f'{gender}_{percentile}_2D.png')

    plt.show()


def calculate_distances(joints):
    foot_x, foot_y = joints[0]
    hand_x, hand_y = joints[6]
    torso_x, torso_y = joints[3]

    # A: Horizontal distance between the free end of the hand and the joint connecting the thigh to the torso
    distance_A = abs(hand_x - torso_x)
    
    # B: Vertical distance between the free end of the foot and the free end of the hand
    distance_B = abs(hand_y - foot_y)

    # C: Vertical distance between the free end of the foot and the joint connecting the thigh to the torso
    distance_C = abs(torso_y - foot_y)
    
    # D: Horizontal distance between the free end of the foot and the joint connecting the thigh to the torso
    distance_D = abs(torso_x - foot_x)

    return distance_A, distance_B, distance_C, distance_D


def generate_samples(data, feature, gender='Man'):
    mu = data[feature][f'{gender}50']
    sigma = (data[feature][f'{gender}95'] - data[feature][f'{gender}5']) / 3.29
    samples = np.random.normal(mu, sigma, 1000)
    return samples