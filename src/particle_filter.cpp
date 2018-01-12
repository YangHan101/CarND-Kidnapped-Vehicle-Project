/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine(gen);

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i)
	{
		double sample_x, sample_y, sample_theta;
		Particle p_;
		p_.x = dist_x(gen);
		p_.y = dist_y(gen);
		p_.theta = dist_theta(gen);
		p_.id = -1;

		particles.push_back(p_);

		// cout << "Added " << i << "th particle: " << p_.x << "\t" << p_.y << "\t" << p_.theta << std::endl;
	}

	//cout << "Initialization done" << '\n';
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine(gen);
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	// std::cout << "dt =" << delta_t << '\n';
	// std::cout << "vel =" << velocity << '\n';
	// std::cout << "yaw_rate =" << yaw_rate << '\n';

	for (int i = 0; i < num_particles; ++i)
	{
		//std::cout << i << "\t"<< particles[i].x<< "\t" << particles[i].y<< "\t" << particles[i].theta  << '\n';
		if (fabs(yaw_rate > 1e-4)) {
			double vel_yaw_ratio = velocity / yaw_rate;
			double theta_new = particles[i].theta + delta_t * yaw_rate;
			particles[i].x += vel_yaw_ratio * (sin(theta_new) - sin(particles[i].theta));
			particles[i].y += vel_yaw_ratio * (cos(particles[i].theta) - cos(theta_new));
			particles[i].theta = theta_new;
		}
		else {
			double dist_travel_ = velocity * delta_t;
			particles[i].x += dist_travel_ * cos(particles[i].theta);
			particles[i].y += dist_travel_ * sin(particles[i].theta);
		}
		// std::cout << i << "\t"<< particles[i].x<< "\t" << particles[i].y<< "\t" << particles[i].theta  << '\n';

		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
		particles[i].weight = 1.0;
		// std::cout << i << "\t"<< particles[i].x<< "\t" << particles[i].y<< "\t" << particles[i].theta  << '\n';

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	int n_pred = predicted.size();
	int n_obs = observations.size();

	for (int i = 0; i < n_pred; i++) { // for every actual landmark
		double shortest_dist = numeric_limits<double>::max();
		int best_obs = -1;
		for (int j = 0; j < n_obs; j++) { // for every predicted landmark location based on particles' view
			double dist_ij = dist(predicted[i].x, predicted[i].y, observations[j].x, observations[j].y);
			// std::cout << "landmark_" << predicted[i].id << " to obs_" << j << ": " << dist_ij << '\n';
			if (dist_ij < shortest_dist) {
				shortest_dist = dist_ij;
				best_obs = j;
			}

		}
		observations[best_obs].id = i;
		// std::cout << "For landmark_" << predicted[i].id << "<==>obs_" << best_obs << '\n';
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double weight_sum = 0.0;
	std::vector<LandmarkObs> obs_in_map;

	for (int p_id = 0; p_id < num_particles; p_id++) {
		Particle p_ = particles[p_id];
		// cout << p_id << "th particle: " << p_.x << "\t" << p_.y << "\t" << p_.theta << endl;
		// cout << "It has " << observations.size() << " observations" << endl;

		obs_in_map.clear();
		double cos_theta = cos(p_.theta);
		double sin_theta = sin(p_.theta);
		// cout << "precompute " << "cos(theta) = " << cos_theta << "\tsin(theta) = " << sin_theta << endl;

		for (int i =0; i < observations.size(); i++) {
			LandmarkObs obs_;
			// cout << "obs_" << i<< ":" << observations[i].x << "\t" << observations[i].y << endl;
			obs_.x = observations[i].x * cos_theta - observations[i].y * sin_theta + p_.x;
			obs_.y = observations[i].x * sin_theta + observations[i].y * cos_theta + p_.y;
			obs_.id = -1;
			// cout << "Cvt to map coord obs_" << i<< ":" << obs_.x << "\t" << obs_.y << endl;

			obs_in_map.push_back(obs_);
		}

		std::vector<LandmarkObs> map_in_range;
		// std::cout << "Below are the landmarks in consideration" << '\n';
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			LandmarkObs landmark_;
			landmark_.x = map_landmarks.landmark_list[j].x_f;
			landmark_.y = map_landmarks.landmark_list[j].y_f;
			landmark_.id = map_landmarks.landmark_list[j].id_i;

			double landmark_dist_ = dist(landmark_.x, landmark_.y, p_.x, p_.y);
			if (landmark_dist_ < sensor_range) {
				map_in_range.push_back(landmark_);
				// cout << "landmark_" << j << ":" << landmark_.x << "\t" << landmark_.y << endl;
			}
		}
		dataAssociation(map_in_range, obs_in_map);
		double prob_particle = 1.0;
		double norm_factor = sqrt(M_PI * 2 * std_landmark[0] * std_landmark[1]);

		for (int i = 0; i < obs_in_map.size(); i++) {
			// std::cout << i << ":\t" << typeid(obs_in_map[i]).name() << '\n';
			LandmarkObs obs_ = obs_in_map[i];
			if (obs_.id != -1) {
				// std::cout << "# of objects in the map:\t" << map_in_range.size() << '\n';
				// std::cout << "Which object? \t" << obs_.id << '\n';

				double x_ = map_in_range[obs_.id].x;
				double y_ = map_in_range[obs_.id].y;

				// std::cout << "dx = " << obs_.x - x_ << '\n';
				// std::cout << "dy = " << obs_.y - y_ << '\n';
				// std::cout << "sx = " << std_landmark[0] << '\n';
				// std::cout << "sy = " << std_landmark[1] << '\n';

				double prob_obs = -0.5 * (pow((obs_.x - x_) / std_landmark[0], 2) + pow((obs_.y - y_) / std_landmark[1], 2));

				prob_particle *= exp(prob_obs) / norm_factor;
				// cout << "landmark_" << obs_.id << "<=> obs_" << i << " with prob of " << exp(prob_obs) << endl;
			}
		}

		particles[p_id].weight = prob_particle;
		weight_sum += prob_particle;
	}

	for (int p_id = 0; p_id < num_particles; p_id++) {
		particles[p_id].weight /= weight_sum;
		// std::cout << "Particle_" << p_id << "\t" << particles[p_id].weight << '\n';
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine(gen);
	weights.clear();
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}
	std::discrete_distribution<> resampler (weights.begin(), weights.end());

	std::vector<Particle> next_particles;
	for (int i = 0; i < num_particles; i++) {
		int p_id = resampler(gen);
		// std::cout << "winner particle: " << p_id << '\n';
		next_particles.push_back(particles[p_id]);
	}
	particles = next_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
