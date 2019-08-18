#include <iostream>
#include <numeric>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "dataStructures.h"
#include "parallel.h"

using namespace std;

void detectObjects_YOLO()
{
    /*
     Prepare the Model
      After loading the network, the DNN backend is set to DNN_BACKEND_OPENCV. 
      If OpenCV is built with Intel’s Inference Engine, DNN_BACKEND_INFERENCE_ENGINE should be used instead. 
      The target is set to CPU in the code, as opposed to using DNN_TARGET_OPENCL, 
      which would be the method of choice if a (Intel) GPU was available.
     */
    // load image from file
    cv::Mat img = cv::imread("../images/0000000000.png");

    // load class names from file
    string yoloBasePath = "../dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights"; 

    vector<string> classes;
    ifstream ifs(yoloClassesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // load neural network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    /*
     Prepare the Model end
     */

    /*
     Generate 4D Blob from Input Image
     The size of the input image is controlled by the parameters ‚inpWidth‘ and ‚inpHeight‘, 
     which is set to 416 as proposed by the YOLO authors. 
     Other values could e.g. be 320 (faster) or 608 (more accurate).
     */

    // generate 4D blob from input image
    cv::Mat blob;
    double scalefactor = 1/255.0;
    cv::Size size = cv::Size(320, 320);
    cv::Scalar mean = cv::Scalar(0,0,0);
    bool swapRB = false;
    bool crop = false;
    cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);
    /*
     Generate 4D Blob from Input Image end
     */
    
    /*
    Run Forward Pass Through the Network
     */
    // Get names of output layers
    vector<cv::String> names;
    vector<int> outLayers = net.getUnconnectedOutLayers(); // get indices of output layers, i.e. layers with unconnected outputs
    vector<cv::String> layersNames = net.getLayerNames(); // get names of all layers in the network
    
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) // Get the names of the output layers in names
    {
        names[i] = layersNames[outLayers[i] - 1];
    }

    // invoke forward propagation through network
    vector<cv::Mat> netOutput;
    net.setInput(blob);
    net.forward(netOutput, names);

    // Scan through all bounding boxes and keep only the ones with high confidence
    float confThreshold = 0.40;
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    for (size_t i = 0; i < netOutput.size(); ++i)
    {
        float* data = (float*)netOutput[i].data;
        for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols)
        {
            cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
            cv::Point classId;
            double confidence;
            
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
            if (confidence > confThreshold)
            {
                cv::Rect box; int cx, cy;
                cx = (int)(data[0] * img.cols);
                cy = (int)(data[1] * img.rows);
                box.width  = (int)(data[2] * img.cols);
                box.height = (int)(data[3] * img.rows);
                box.x = cx - box.width/2; // left
                box.y = cy - box.height/2; // top
                
                boxes.emplace_back(box);
                classIds.emplace_back(classId.x);
                confidences.emplace_back((float)confidence);
            }
        }
    }

    /*
    Run Forward Pass Through the Network end
     */

    /*
    Post-Processing of Network Output
     */

    // perform non-maxima suppression
    /*
    The Non Maximum Suppression is controlled by the nmsThreshold parameter. 
    If nmsThreshold is set too low, e.g. 0.1, we might not detect overlapping objects of 
    same or different classes. But if it is set too high e.g. 1, then we get multiple boxes 
    for the same object. So we used an intermediate value of 0.4.
     */
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    std::vector<BoundingBox> bBoxes;
    for (auto it = indices.begin(); it != indices.end(); ++it)
    {
        BoundingBox bBox;
        bBox.roi = boxes[*it];
        bBox.classID = classIds[*it];
        bBox.confidence = confidences[*it];
        bBox.boxID = (int)bBoxes.size(); // zero-based unique identifier for this bounding box
        
        bBoxes.emplace_back(bBox);
    }
    /*
    Post-Processing of Network Output end
     */
    
    // show results
    cv::Mat visImg = img.clone();
    for (auto it = bBoxes.begin(); it != bBoxes.end(); ++it)
    {
        // Draw rectangle displaying the bounding box
        int top, left, width, height;
        top = (*it).roi.y;
        left = (*it).roi.x;
        width = (*it).roi.width;
        height = (*it).roi.height;
        cv::rectangle(visImg, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);

        string label = cv::format("%.2f", (*it).confidence);
        label = classes[((*it).classID)] + ":" + label;

        // Display label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        rectangle(visImg, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0, 0, 0), 1);
    }

    string windowName = "Object classification";
    cv::namedWindow( windowName, 1 );
    cv::imshow( windowName, visImg );
    cv::waitKey(0); // wait for key to be pressed
}

void detectObjects_YOLO_parallel()
{
    /*
     Prepare the Model
      After loading the network, the DNN backend is set to DNN_BACKEND_OPENCV. 
      If OpenCV is built with Intel’s Inference Engine, DNN_BACKEND_INFERENCE_ENGINE should be used instead. 
      The target is set to CPU in the code, as opposed to using DNN_TARGET_OPENCL, 
      which would be the method of choice if a (Intel) GPU was available.
     */
    // load image from file
    cv::Mat img = cv::imread("../images/0000000000.png");

    // load class names from file
    string yoloBasePath = "../dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights"; 

    vector<string> classes;
    ifstream ifs(yoloClassesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.emplace_back(line);
    
    // load neural network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    /*
     Prepare the Model end
     */

    /*
     Generate 4D Blob from Input Image
     The size of the input image is controlled by the parameters ‚inpWidth‘ and ‚inpHeight‘, 
     which is set to 416 as proposed by the YOLO authors. 
     Other values could e.g. be 320 (faster) or 608 (more accurate).
     */

    // generate 4D blob from input image
    cv::Mat blob;
    double scalefactor = 1/255.0;
    cv::Size size = cv::Size(320, 320);
    cv::Scalar mean = cv::Scalar(0,0,0);
    bool swapRB = false;
    bool crop = false;
    cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);
    /*
     Generate 4D Blob from Input Image end
     */
    
    /*
    Run Forward Pass Through the Network
     */
    // Get names of output layers
    vector<cv::String> names;
    vector<int> outLayers = net.getUnconnectedOutLayers(); // get indices of output layers, i.e. layers with unconnected outputs
    vector<cv::String> layersNames = net.getLayerNames(); // get names of all layers in the network
    
    names.resize(outLayers.size());
    // Get the names of the output layers in names
    parallel_for (outLayers.size(), [&] (int start, int end)
    {
       for (size_t i = 0; i < outLayers.size(); ++i)
       {
           names[i] = layersNames[outLayers[i] - 1];
       }
    });

    // invoke forward propagation through network
    vector<cv::Mat> netOutput;
    net.setInput(blob);
    net.forward(netOutput, names);

    // Scan through all bounding boxes and keep only the ones with high confidence
    float confThreshold = 0.40;
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    std::mutex m;
    parallel_for (netOutput.size(), [&] (int start, int end)
    {
        for (size_t i = 0; i < netOutput.size(); ++i)
        {
            float* data = (float*)netOutput[i].data;
            for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols)
            {
                cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
                cv::Point classId;
                double confidence;
                
                // Get the value and location of the maximum score
                cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
                if (confidence > confThreshold)
                {
                    cv::Rect box; int cx, cy;
                    cx = (int)(data[0] * img.cols);
                    cy = (int)(data[1] * img.rows);
                    
                    box.width  = (int)(data[2] * img.cols);
                    box.height = (int)(data[3] * img.rows);
                    box.x = cx - box.width/2; // left
                    box.y = cy - box.height/2; // top

                    std::lock_guard<std::mutex> lk(m);
                    boxes.emplace_back(box);
                    classIds.emplace_back(classId.x);
                    confidences.emplace_back((float)confidence);
            }
        }
    }
    });

    /*
    Run Forward Pass Through the Network end
     */

    /*
    Post-Processing of Network Output
     */

    // perform non-maxima suppression
    /*
    The Non Maximum Suppression is controlled by the nmsThreshold parameter. 
    If nmsThreshold is set too low, e.g. 0.1, we might not detect overlapping objects of 
    same or different classes. But if it is set too high e.g. 1, then we get multiple boxes 
    for the same object. So we used an intermediate value of 0.4.
     */
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    std::vector<BoundingBox> bBoxes;

    std::mutex m2;
    parallel_for (indices.size(), [&] (int start, int end)
    {
       for (size_t i = 0; i < indices.size(); ++i)
       {
           BoundingBox bBox;
           bBox.roi = boxes[indices[i]];
           bBox.classID = classIds[indices[i]];
           bBox.confidence = confidences[indices[i]];
           bBox.boxID = (int)bBoxes.size(); // zero-based unique identifier for this bounding box
           
           std::lock_guard<std::mutex> lk(m2);
           bBoxes.emplace_back(bBox);
       }
    });
    
    /*
    Post-Processing of Network Output end
     */
    
    // show results
    cv::Mat visImg = img.clone();
    parallel_for (bBoxes.size(), [&] (int start, int end)
    {
       for (size_t i = 0; i < bBoxes.size(); ++i)
       {
           // Draw rectangle displaying the bounding box
           int top, left, width, height;
           top = bBoxes[i].roi.y;
           left = bBoxes[i].roi.x;
           width = bBoxes[i].roi.width;
           height = bBoxes[i].roi.height;
           cv::rectangle(visImg, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);
           
           string label = cv::format("%.2f", bBoxes[i].confidence);
           label = classes[(bBoxes[i].classID)] + ":" + label;
           
           // Display label at the top of the bounding box
           int baseLine;
           cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
           top = max(top, labelSize.height);
           rectangle(visImg, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
           cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0, 0, 0), 1);
           
       }
    });


    string windowName = "Object classification";
    cv::namedWindow( windowName, 1 );
    cv::imshow( windowName, visImg );
    cv::waitKey(0); // wait for key to be pressed
}

void detectObjects_mobile_SSD()
{
    
    /*
     Prepare the Model
     */
    // load image from file
    cv::Mat img = cv::imread("../images/0000000000.png");
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << std::endl;
        exit(-1);
    }

    // load class names from file
    string ssd_BasePath = "../dat/mobilenet_ssd/"; 
    string modelTxt = ssd_BasePath+ "MobileNetSSD_deploy.prototxt.txt";
    string modelBin = ssd_BasePath+ "MobileNetSSD_deploy.caffemodel";

    string CLASSES[] = {"background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"};
    
    // load neural network
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
    net.setPreferableTarget(1);
    if (net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        exit(-1);
    }
    /*
     Prepare the Model end
     */

    /*
    Generate 4D blob from input image
     MobileNet requires fixed dimensions for input image(s)
     so we have to ensure that it is resized to 300x300 pixels.
     set a scale factor to image because network the objects has differents size. 
     We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
     after executing this command our "blob" now has the shape:
     (1, 3, 300, 300) 
     */

    cv::Mat img2;
    resize(img, img2, cv::Size(300,300));
    cv::Mat inputBlob = cv::dnn::blobFromImage(img2, 0.007843, cv::Size(300,300), 
                                               cv::Scalar(127.5, 127.5, 127.5), false);
    //Set to network the input blob 
    net.setInput(inputBlob, "data");

    /*
     Generate 4D Blob from Input Image end
     */
    
    /*
    Run Forward Pass Through the Network
     */
    
    // invoke forward propagation through network
    //double time = (double)cv::getTickCount();
    cv::Mat detection = net.forward("detection_out"); 
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    //double time_elpased = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    //std::cout<<"time spend: "<<time_elpased<<std::endl;


    // Scan through all bounding boxes and keep only the ones with high confidence

    float confThreshold = 0.4;
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confThreshold)
        {
            int idx = static_cast<int>(detectionMat.at<float>(i, 1));
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

            cv::Rect box((int)xLeftBottom, (int)yLeftBottom,
                         (int)(xRightTop - xLeftBottom),
                         (int)(yRightTop - yLeftBottom));
            boxes.emplace_back(box);
            classIds.emplace_back(idx);
            confidences.emplace_back(confidence);

            //cout << CLASSES[idx] << ": " << confidence << endl;
        }
    }

    /*
    Run Forward Pass Through the Network end
     */

    /*
    Post-Processing of Network Output
     */

    // perform non-maxima suppression
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    std::vector<BoundingBox> bBoxes;
    for (auto it = indices.begin(); it != indices.end(); ++it)
    {
        BoundingBox bBox;
        bBox.roi = boxes[*it];
        bBox.classID = classIds[*it];
        bBox.confidence = confidences[*it];
        bBox.boxID = (int)bBoxes.size(); // zero-based unique identifier for this bounding box
        
        bBoxes.emplace_back(bBox);
    }

    /*
    Post-Processing of Network Output end
     */
    
    // show results
    cv::Mat visImg = img.clone();
    for (auto it = bBoxes.begin(); it != bBoxes.end(); ++it)
    {
        // Draw rectangle displaying the bounding box
        int top, left, width, height;
        top = (*it).roi.y;
        left = (*it).roi.x;
        width = (*it).roi.width;
        height = (*it).roi.height;
        cv::rectangle(visImg, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);

        string label = cv::format("%.2f", (*it).confidence);
        label = CLASSES[((*it).classID)] + ":" + label;

        // Display label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        rectangle(visImg, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(visImg, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
    }

    string windowName = "Object classification";
    cv::namedWindow( windowName, 1 );
    cv::imshow( windowName, visImg );
    cv::waitKey(0); // wait for key to be pressed
}

void detectObjects_mobile_SSD_parallel()
{
    
    /*
     Prepare the Model
     */
    // load image from file
    cv::Mat img = cv::imread("../images/0000000000.png");
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << std::endl;
        exit(-1);
    }

    // load class names from file
    string ssd_BasePath = "../dat/mobilenet_ssd/"; 
    string modelTxt = ssd_BasePath+ "MobileNetSSD_deploy.prototxt.txt";
    string modelBin = ssd_BasePath+ "MobileNetSSD_deploy.caffemodel";

    string CLASSES[] = {"background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"};
    
    // load neural network
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
    net.setPreferableTarget(1);
    if (net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        exit(-1);
    }
    /*
     Prepare the Model end
     */

    /*
    Generate 4D blob from input image
     MobileNet requires fixed dimensions for input image(s)
     so we have to ensure that it is resized to 300x300 pixels.
     set a scale factor to image because network the objects has differents size. 
     We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
     after executing this command our "blob" now has the shape:
     (1, 3, 300, 300) 
     */

    cv::Mat img2;
    resize(img, img2, cv::Size(300,300));
    cv::Mat inputBlob = cv::dnn::blobFromImage(img2, 0.007843, cv::Size(300,300), 
                                               cv::Scalar(127.5, 127.5, 127.5), false);
    //Set to network the input blob 
    net.setInput(inputBlob, "data");

    /*
     Generate 4D Blob from Input Image end
     */
    
    /*
    Run Forward Pass Through the Network
     */
    
    // invoke forward propagation through network
    //double time = (double)cv::getTickCount();
    cv::Mat detection = net.forward("detection_out"); 
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    //double time_elpased = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    //std::cout<<"time spend: "<<time_elpased<<std::endl;


    // Scan through all bounding boxes and keep only the ones with high confidence

    float confThreshold = 0.2;
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    std::mutex m;
    parallel_for (detectionMat.rows, [&] (int start, int end)
    {
       for (size_t i = 0; i < detectionMat.rows; ++i)
       {
           float confidence = detectionMat.at<float>(i, 2);
           
           if (confidence > confThreshold)
           {
               int idx = static_cast<int>(detectionMat.at<float>(i, 1));
               int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
               int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
               int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
               int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);
               
               cv::Rect box((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));
                            
               std::lock_guard<std::mutex> lk(m);
               boxes.emplace_back(box);
               classIds.emplace_back(idx);
               confidences.emplace_back(confidence);
               //cout << CLASSES[idx] << ": " << confidence << endl;
        }
       }
    });

    /*
    Run Forward Pass Through the Network end
     */

    /*
    Post-Processing of Network Output
     */

    // perform non-maxima suppression
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    std::vector<BoundingBox> bBoxes;
    
    std::mutex m2;
    parallel_for (indices.size(), [&] (int start, int end)
    {
       for (size_t i = 0; i < indices.size(); ++i)
       {
           BoundingBox bBox;
           bBox.roi = boxes[indices[i]];
           bBox.classID = classIds[indices[i]];
           bBox.confidence = confidences[indices[i]];
           bBox.boxID = (int)bBoxes.size(); // zero-based unique identifier for this bounding box
           
           std::lock_guard<std::mutex> lk(m2);
           bBoxes.emplace_back(bBox);
       }
    });


    /*
    Post-Processing of Network Output end
     */
    
    // show results
    cv::Mat visImg = img.clone();
    for (auto it = bBoxes.begin(); it != bBoxes.end(); ++it)
    {
        // Draw rectangle displaying the bounding box
        int top, left, width, height;
        top = (*it).roi.y;
        left = (*it).roi.x;
        width = (*it).roi.width;
        height = (*it).roi.height;
        cv::rectangle(visImg, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);

        string label = cv::format("%.2f", (*it).confidence);
        label = CLASSES[((*it).classID)] + ":" + label;

        // Display label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        rectangle(visImg, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(visImg, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
    }

    string windowName = "Object classification";
    cv::namedWindow( windowName, 1 );
    cv::imshow( windowName, visImg );
    cv::waitKey(0); // wait for key to be pressed
}

int main()
{
    double time = (double)cv::getTickCount();
    std::cout<<"YOLO: " <<std::endl;
    detectObjects_YOLO_parallel();
    double time_elpased = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    std::cout<<"time spend on YOLO: "<<time_elpased<<std::endl;
    
    time = (double)cv::getTickCount();
    std::cout<<"SSD: " <<std::endl;
    detectObjects_mobile_SSD_parallel();
    time_elpased = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    std::cout<<"time spend on SSD: "<<time_elpased<<std::endl;
}