import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from scipy.stats import norm


def calc_angles(angles_in):
    # Takes joint angles -> outputs angles relative to x-axis
    # angles_in = [45, 45, 110, 30, 135]

    a = angles_in[0]
    b = 180 - (angles_in[1] - a)
    c = angles_in[2] - (180 - b)
    d = 180 + c + angles_in[3]
    e = d - angles_in[4] -180

    angles_out = [180, a, b, c, d, e]

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


def plot_skeleton(joints, dists = True, gender = 'Man', percentile = '50'):
    
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
        # Horizontal A
        # ax.plot([joint_x[3], joint_x[6]], [joint_y[4] + 250,joint_y[4] + 250], color='black')
        ax.annotate('', 
            xy=(joint_x[3], joint_y[4] + 250),  # Start of the line (xy)
            xytext=(joint_x[6], joint_y[4] + 250),  # End of the line (xytext)
            arrowprops=dict(facecolor='black', arrowstyle='<-'))  # Arrow at the start
        ax.annotate('', 
            xy=(joint_x[3], joint_y[4] + 250),  # Start of the line (xy)
            xytext=(joint_x[6], joint_y[4] + 250),  # End of the line (xytext)
            arrowprops=dict(facecolor='black', arrowstyle='->'))  # Arrow at the end
        ax.vlines(joint_x[3], joint_y[3], joint_y[4]+250 , linestyle='--', colors='black', linewidth=0.5)
        ax.vlines(joint_x[6], joint_y[6], joint_y[4]+250 , linestyle='--', colors='black', linewidth=0.5)
        ax.text((joint_x[3] + joint_x[6]) / 2, joint_y[4] + 260, "Distance A", ha='center', va='bottom', fontsize=10)

        # Vertical B
        # ax.plot([joint_x[6] + 250, joint_x[6] + 250] , [joint_y[0], joint_y[6]], color='black')
        ax.annotate('', 
            xy=(joint_x[6] + 250, joint_y[0]),  # Start of the line (xy)
            xytext=(joint_x[6] + 250, joint_y[6]),  # End of the line (xytext)
            arrowprops=dict(facecolor='black', arrowstyle='<-'))  # Arrow at the start
        ax.annotate('', 
              xy=(joint_x[6] + 250, joint_y[0]),  # Start of the line (xy)
            xytext=(joint_x[6] + 250, joint_y[6]),  # End of the line (xytext)
            arrowprops=dict(facecolor='black', arrowstyle='->'))  # Arrow at the end
        ax.hlines(joint_y[0], joint_x[0], joint_x[6]+250 , linestyle='--', colors='black', linewidth=0.5)
        ax.hlines(joint_y[6], joint_x[6], joint_x[6]+250 , linestyle='--', colors='black', linewidth=0.5)
        ax.text(joint_x[6] + 280, (joint_y[0] + joint_y[6]) / 2, "Distance B", ha='left', va='center', fontsize=10, rotation=90)

    # Adjust the plot limits dynamically to fit all joints
    x_min = min(joint_x) - 300
    x_max = max(joint_x) + 500
    y_min = min(joint_y) - 100
    y_max = max(joint_y) + 400
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.title(f"2D Human Skeleton, {gender}, {percentile}%")
    plt.show()


def calculate_distances(joints):
    # A: Horizontal distance between the free end of the hand and the joint connecting the thigh to the torso
    hand_x, hand_y = joints[6]
    torso_x, torso_y = joints[3]
    distance_A = abs(hand_x - torso_x)
    
    # B: Vertical distance between the free end of the foot and the free end of the hand
    foot_x, foot_y = joints[0]
    distance_B = abs(hand_y - foot_y)
    
    return distance_A, distance_B


def generate_samples(data, feature, gender='Man'):
    mu = data[feature][f'{gender}50']
    sigma = (data[feature][f'{gender}95'] - data[feature][f'{gender}5']) / 3.29
    samples = np.random.normal(mu, sigma, 1000)
    return samples