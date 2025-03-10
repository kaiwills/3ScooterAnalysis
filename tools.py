import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from scipy.stats import norm

def calculate_joint_positions(dimensions, angles):
    # Initialize the list of joints with the first joint (origin of the foot) at the origin
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


#######

def calc_A(df, gender='Man'):
    
    df['TorsoHeight'][f'{gender}95']  * df['Forearm'][f'{gender}95'] 
    
    for i in df.cols:
        i['Man95'] 


def calc_B(df):



#######



def plot_skeleton(joints):
    # Plot the skeleton using the joints
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    
    # Plot the body parts as lines between joints
    for i in range(1, len(joints)):
        x_values = [joints[i-1][0], joints[i][0]]
        y_values = [joints[i-1][1], joints[i][1]]
        ax.plot(x_values, y_values, color='blue')  # Plot lines between joints
    
    # Plot dots at each joint
    joint_x, joint_y = zip(*joints)  # Unzip the x and y coordinates of the joints
    ax.scatter(joint_x, joint_y, color='red')  # Plot the joints as dots
    
    # Adjust the plot limits dynamically to fit all joints
    x_min = min(joint_x) - 1
    x_max = max(joint_x) + 1
    y_min = min(joint_y) - 1
    y_max = max(joint_y) + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.title("2D Human Skeleton (Seated Position)")
    plt.show()
 
 
    
def generate_samples(data, feature, gender='Man'):
    mu = data[feature][f'{gender}50']
    sigma = (data[feature][f'{gender}95'] - data[feature][f'{gender}5']) / 3.29
    samples = np.random.normal(mu, sigma, 1000)
    return samples