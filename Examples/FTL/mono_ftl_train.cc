/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include <unistd.h>

#include<opencv2/core/core.hpp>

#include "System.h"
#include "Converter.h"


void LoadImages(const std::string &databasePath,
                std::vector<std::string> &vstrImages, std::vector<double> &vTimeStamps)
{
    std::ifstream fTimes(databasePath + "/database_entries.csv");
    
    // Read header
    std::string header;
    std::getline(fTimes, header);
    
    // Parse
    int filenameIdx = -1;
    int timestepIdx = -1;
    std::istringstream headerStream(header);
    std::string headerField;
    for (int i = 0; std::getline(headerStream, headerField, ','); i++) {
        if(headerField == " Timestamp [ms]") {
            timestepIdx = i;
        }
        else if(headerField == " Filename") {
            filenameIdx = i;
        }
    }
    
    assert(timestepIdx != -1 && filenameIdx != -1);
    
    // Get lines
    for(std::string line; std::getline(fTimes, line);) {
        // Parse lines
        std::istringstream lineStream(line);
        std::string lineField;
        for (int i = 0; std::getline(lineStream, lineField, ','); i++) {
            // Convert timestep to seconds
            if(i == timestepIdx) {
                vTimeStamps.push_back(std::stod(lineField) / 1000.0);
            }
            // Add path an persp suffix
            else if(i == filenameIdx) {
                vstrImages.push_back(databasePath + "/" + lineField.substr(1, lineField.size() - 5) + "_persp..jpg");
            }
        }
        assert(vTimeStamps.size() == vstrImages.size());
    }
}

int main(int argc, char **argv)
{
    /*bool bFileName= (((argc-3) % 2) == 1);

    std::string file_name;
    if (bFileName)
    {
        file_name = std::string(argv[argc-1]);
        cout << "file name: " << file_name << endl;
    }*/


    if(argc < 3)
    {
        cerr << endl << "Usage: ./mono_ftl path_to_vocabulary path_to_settings path_to_database_folder" << endl;
        return 1;
    }

    // Load all sequences:
    std::vector<std::string> vstrImageFilenames;
    std::vector<double> vTimestampsCam;

    std::cout << "Loading database...";
    LoadImages(std::string(argv[3]), vstrImageFilenames, vTimestampsCam);
    std::cout << "LOADED!" << endl;

    const int nImages = vstrImageFilenames.size();
    const int tot_images = nImages;

    if((nImages<=0))
    {
        std::cerr << "ERROR: Failed to load images " << endl;
        return 1;
    }

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    std::cout << std::endl << "-------" << std::endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    const bool visualise = true;
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, visualise, 0);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;
    double ttrack_tot = 0;

    // Main loop
    cv::Mat im;
    int proccIm = 0;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    for(int ni=0; ni<nImages; ni++, proccIm++)
    {

        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],cv::IMREAD_GRAYSCALE); //,cv::IMREAD_GRAYSCALE);

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
#endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
#endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

        // clahe
        clahe->apply(im,im);


        // cout << "mat type: " << im.type() << endl;
        double tframe = vTimestampsCam[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                    <<  vstrImageFilenames[ni] << endl;
            return 1;
        }
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe); // TODO change to monocular_inertial

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        ttrack_tot += ttrack;

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestampsCam[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestampsCam[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6); // 1e6

    }

    // cout << "ttrack_tot = " << ttrack_tot << std::endl;
    // Stop all threads
    SLAM.Shutdown();


    // Tracking time statistics

    // Save camera trajectory

    /*if (bFileName)
    {
        const std::string kf_file =  "kf_" + std::string(argv[argc-1]) + ".txt";
        const std::string f_file =  "f_" + std::string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {*/
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    //}

    std::sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    std::cout << "-------" << std::endl << std::endl;
    std::cout << "median tracking time: " << vTimesTrack[nImages/2] << std::endl;
    std::cout << "mean tracking time: " << totaltime/proccIm << std::endl;


    return 0;
}
