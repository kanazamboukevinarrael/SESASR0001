
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, degrees
import matplotlib as mpl 

arrow = u'$\u2191$'

def evaluate_sampling_dist(mu, sigma, n_samples, sample_function, n_bins=100):
    """ 
    Evaluate the sampling distribution by generating samples, 
    computing mean and std dev, and plotting a histogram. 
    """
    samples = []

    for i in range(n_samples):
        samples.append(sample_function(mu, sigma))

    #print("%30s : mean = %.3f, std_dev = %.3f" % ("Normal", np.mean(samples), np.std(samples))) ### to do non have them on the node

    count, bins, ignored = plt.hist(samples, n_bins)
    # plt.plot(bins, norm(mu, sigma).pdf(bins), linewidth=2, color='r')
    plt.xlim([mu - 5*sigma, mu + 5*sigma])
    plt.title("Normal distribution of samples")
    plt.grid()
    # plt.savefig("velocity_model_gaussian_dist.pdf")
    #plt.show() ### because it is apparing on the node where
    #  it is clalled samme for all print that are here presente here  
    # they were need to evalute if the code was working but there are 
    # non longer needed, so i commented them out to avoid the pop up of the plot and the print on the terminal



def sample_velocity_motion_model(x, u, a, dt):
    """ 
    Sample velocity motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- velocity reading obtained from the robot [v, w]
    a -- noise parameters of the motion model [a1, a2, a3, a4, a5, a6]
    dt -- time interval of prediction
    """
    v_hat = u[0] + np.random.normal(0, a[0]*u[0]**2 + a[1]*u[1]**2)
    w_hat = u[1] + np.random.normal(0, a[2]*u[0]**2 + a[3]*u[1]**2)
    gamma_hat = np.random.normal(0, a[4]*u[0]**2 + a[5]*u[1]**2)

    r = v_hat/w_hat
    x_prime = x[0] - r*sin(x[2]) + r*sin(x[2]+w_hat*dt)
    y_prime = x[1] + r*cos(x[2]) - r*cos(x[2]+w_hat*dt)
    theta_prime = x[2] + w_hat*dt + gamma_hat*dt

    the_array = np.array([x_prime, y_prime, theta_prime])

    return np.array([x_prime, y_prime, theta_prime]) ### This turn a list of probable positions givent the cmd and the noise



def heart_velocity_model(x, u, a):
    plt.close('all')
    n_samples = 500
    dt = 0.5

    x = x # initial pose
    u = u # velocity command (linear and angular)
    a = a # noise variance

    x_prime = np.zeros([n_samples, 3])
    for i in range(n_samples):
        x_prime[i,:] = sample_velocity_motion_model(x, u, a, dt)

    ###################################
    ######### Plot x samples ##########
    ###################################
       
    mu = np.mean(x_prime, axis=0)
    sigma = np.std(x_prime, axis=0)
    #evaluate_sampling_dist(mu[0], sigma[0], n_samples, np.random.normal) ### commented because we need the pose nbont the guassian distribution

    ###################################
    ### Sampling the velocity model ###
    ###################################

    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x[2])-90)
    plt.scatter(x[0], x[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')

    for x_ in x_prime[:200]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
        plt.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='r')

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("velocity motion model sampling")
    # plt.savefig("velocity_samples.pdf")
    #plt.show() ### because we don't need the polt os single calculation
    plt.close() # close the plot to avoid memory issues

    ###################################
    #### Multiple steps of sampling ###
    ###################################

    x = x

    noise_list = [[0.1, 0.2, 0.1, 0.2, 0.05, 0.05], a, [0.2, 0.4, 0.2, 0.4, 0.1, 0.1]] # different noise levels

    #a = [0.05, 0.1, 0.05, 0.1, 0.025, 0.025] # noise variance
    cmds = [
        [0.8, 0]] ### i feed him just one velocity command because we will the ges just the next prose out of the probabiistic calculation,
                    ### if we want to see the next pose we need to feed him the next velocity command, so we can see the next pose, and so on.
    
    
    output_dir = Path(__file__).resolve().parents[1] / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)   ### to plot all togethe the comparizon os the output from 3 different noise: figures: 1 row and 3 columns
   
    possible_pose =[]
    for a in noise_list:
        x_prime = np.zeros([n_samples, 3])
        for t, u in enumerate(cmds):
            for i in range(0, n_samples):
                x_ = x_prime[i,:]
                if t ==0:
                    x_prime[i,:] = sample_velocity_motion_model(x, u, a, dt)
                else:
                    x_prime[i,:] = sample_velocity_motion_model(x_, u, a, dt)
            
            ax[noise_list.index(a)].plot(x_prime[:,0], x_prime[:,1], "r,")
            ax[noise_list.index(a)].plot(x[0], x[1], "bo", label="previous") ### this is the previous pose used to calculate the sample_velocity_motion_model 
            
            x = np.mean(x_prime, axis=0)
            sigma = np.std(x_prime, axis=0)
            #print("muka: ", x, "sigma: ", sigma)  ### to do non have them on the node

        possible_pose.append(x)
        ax[noise_list.index(a)].plot(x[0], x[1], "ko", label="actual") ### i added this line beacause i need to 
                                ### see the mean position claculated on the plot too
        ax[noise_list.index(a)].set_xlabel("x-position [m]")
        ax[noise_list.index(a)].set_ylabel("y-position [m]")
        ax[noise_list.index(a)].set_title("velocity multiple sampling")
        ax[noise_list.index(a)].legend()
        # plt.savefig("multi_velocity_samples.pdf")
        #ax[noise_list.index(a)].show()

    #ax[noise_list.index(a)].close('all')
    fig.tight_layout()
    fig.savefig(output_dir / "multi_velocity_samples.png") ### i added this line to save the figure with all the comparizon of the 3 different noise levels
    #plt.show()
    

    return possible_pose[1] ### this is the mean position calculated by the probabilistic motion model,
            ### which is the next pose of the robot given the previous pose and the velocity command



if __name__ == "__main__":
    sf = heart_velocity_model(
        x=[2, 4, 0],
        u=[0.8, 0],
        a=[0.05, 0.1, 0.05, 0.1, 0.025, 0.025],
    )
    #print("The next pose calculated by the probabilistic motion model is: ", sf) ### to do non have them on the node

