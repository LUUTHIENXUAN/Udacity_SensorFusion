/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting

#include "../../render/render.h"
#include <unordered_set>
#include "../../processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "../../processPointClouds.cpp"
#include "parallel.h"


pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData2D()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  	// Add inliers
  	float scatter = 0.6;
  	for(int i = -50; i < 50; i++)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = i+scatter*rx;
  		point.y = i+scatter*ry;
  		point.z = 0;

  		cloud->points.push_back(point);
  	}
  	// Add outliers
  	int numOutliers = 40;
  	while(numOutliers--)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = 20*rx;
  		point.y = 20*ry;
  		point.z = 0;

  		cloud->points.push_back(point);

  	}
  	cloud->width = cloud->points.size();
  	cloud->height = 1;

  	return cloud;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData3D()
{
	ProcessPointClouds<pcl::PointXYZ> pointProcessor;
	return pointProcessor.loadPcd("../../../sensors/data/pcd/simpleHighway.pcd");
}

pcl::visualization::PCLVisualizer::Ptr initScene()
{
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("2D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
  	viewer->initCameraParameters();
  	viewer->setCameraPosition(0, 0, 15, 0, 1, 0);
  	viewer->addCoordinateSystem (1.0);
  	return viewer;
}

std::unordered_set<int> Ransac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	// Time segmentation process
	auto startTime = std::chrono::steady_clock::now();
	
	std::unordered_set<int> inliersResult;
	srand(time(NULL));
	
	// TODO: Fill in this function

	// For max iterations 

	// Randomly sample subset and fit line

	// Measure distance between every point and fitted line
	// If distance is smaller than threshold count it as inlier

	// Return indicies of inliers from fitted line with most inliers
	while (maxIterations --)
	{
		// Randomly pick two points
		std::unordered_set<int> inliers;
		while (inliers.size()< 2)
		    inliers.insert(rand()%(cloud->points.size()));

		float x1, y1, x2, y2;

		auto itr = inliers.begin();
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;

		float a = (y1-y2);
		float b = (x2-x1);
		float c = (x1*y2 - x2*y1);

		for(int index = 0; index < cloud->points.size(); index++)
		{
			if (inliers.count(index)>0) continue;

			pcl::PointXYZ point = cloud->points[index];
			float x3 = point.x;
			float y3 = point.y;

			float d = fabs(a*x3 + b*y3 + c)/sqrt(a*a + b*b);
			if (d < distanceTol) inliers.insert(index);
			
		}

		if (inliers.size() > inliersResult.size())
		{
			inliersResult = inliers;
		}

	}

	auto endTime = std::chrono::steady_clock::now();
	auto elapseTime = std::chrono::duration_cast<std::chrono::microseconds> (endTime - startTime);
	std::cout << "Ransac took " << elapseTime.count() << " microseconds" << std::endl;

	return inliersResult;

}

std::unordered_set<int> Ransac2D_earlystop(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	// Time segmentation process
	auto startTime = std::chrono::steady_clock::now();
	
	std::unordered_set<int> inliersResult;
	srand(time(NULL));

	// For max iterations 

	// Randomly sample subset and fit line

	// Measure distance between every point and fitted line
	// If distance is smaller than threshold count it as inlier

	// Return indicies of inliers from fitted line with most inliers
	while (maxIterations --)
	{
		//http://people.inf.ethz.ch/pomarc/pubs/RaguramPAMI13.pdf
		// Randomly pick two points
		std::unordered_set<int> inliers;
		std::unordered_set<int> samples_tried;
		while (inliers.size()< 2)
		{
			int rnd = rand() % cloud->points.size();
			if (samples_tried.count(rnd) >0 ) continue;
			samples_tried.insert(rnd);

			float x_rnd = cloud->points[rnd].x;
			float y_rnd = cloud->points[rnd].y;


			std::unordered_set<int> point_filtered;
			point_filtered.insert(rnd);
			std::cout << "First point: " << rnd << "/" << point_filtered.size()<< std::endl;


			for(int index = 0; index < cloud->points.size(); index++)
			{
				if (point_filtered.count(index)>0) continue;

				pcl::PointXYZ point_spl = cloud->points[index];
				float x_spl = point_spl.x;
				float y_spl = point_spl.y;

				float spatial_x = (x_rnd - x_spl)*(x_rnd - x_spl);
				float spatial_y = (y_rnd - y_spl)*(y_rnd - y_spl);
				if ((spatial_x > 1) || (spatial_y > 1)) continue;

				float spatial   = spatial_x + spatial_y;
				if (spatial < 1.0)
				{
					std::cout << "Spatial: " << spatial << std::endl;
					point_filtered.insert(index);
					std::cout << "Current point: " << point_filtered.size() << std::endl;

					if (point_filtered.size() > 3) 
					{
						std::cout << "total point: " << point_filtered.size() << std::endl;
						inliers.insert(rnd); 
						inliers.insert(rand() % point_filtered.size());

						break;
					}

				}
			}
		}
		    

		float x1, y1, x2, y2;

		auto itr = inliers.begin();
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;

		float a = (y1-y2);
		float b = (x2-x1);
		float c = (x1*y2 - x2*y1);

		float new_distanceTol = distanceTol * sqrt(a*a + b*b);
		int inliersTol = cloud->points.size() * 0.6;

		for(int index = 0; index < cloud->points.size(); index++)
		{
			if (inliers.count(index)>0) continue;

			pcl::PointXYZ point = cloud->points[index];
			float x3 = point.x;
			float y3 = point.y;

			float d = fabs(a*x3 + b*y3 + c);
			if (d < new_distanceTol) inliers.insert(index);
			
		}

		if (inliers.size() > inliersResult.size())
		{
			inliersResult = inliers;
		}

		if (inliers.size() > inliersTol) 
		{
			std::cout << "Early stopped at: " << maxIterations << std::endl;
			break;
		}

	}

	auto endTime = std::chrono::steady_clock::now();
	auto elapseTime = std::chrono::duration_cast<std::chrono::microseconds> (endTime - startTime);
	std::cout << "Ransac took " << elapseTime.count() << " microseconds" << std::endl;

	return inliersResult;

}

std::unordered_set<int> Ransac2D(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	// Time segmentation process
	auto startTime = std::chrono::steady_clock::now();
	
	std::unordered_set<int> inliersResult;
	srand(time(NULL));

	// For max iterations 

	// Randomly sample subset and fit line

	// Measure distance between every point and fitted line
	// If distance is smaller than threshold count it as inlier

	// Return indicies of inliers from fitted line with most inliers
	while (maxIterations --)
	{
		
		// Randomly pick two points
		std::unordered_set<int> inliers;
		while (inliers.size()< 2)
		    inliers.insert(rand()%(cloud->points.size()));	    

		float x1, y1, x2, y2;

		auto itr = inliers.begin();
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;

		float a = (y1-y2);
		float b = (x2-x1);
		float c = (x1*y2 - x2*y1);

		float new_distanceTol = distanceTol * sqrt(a*a + b*b);
		int inliersTol = cloud->points.size() * 0.6;

		for(int index = 0; index < cloud->points.size(); index++)
		{
			if (inliers.count(index)>0) continue;

			pcl::PointXYZ point = cloud->points[index];
			float x3 = point.x;
			float y3 = point.y;

			float d = fabs(a*x3 + b*y3 + c);
			if (d < new_distanceTol) inliers.insert(index);
			
		}

		if (inliers.size() > inliersResult.size())
		{
			inliersResult = inliers;
		}

		if (inliers.size() > inliersTol) 
		{
			std::cout << "Early stopped at: " << maxIterations << std::endl;
			break;
		}

	}

	auto endTime = std::chrono::steady_clock::now();
	auto elapseTime = std::chrono::duration_cast<std::chrono::microseconds> (endTime - startTime);
	std::cout << "Ransac took " << elapseTime.count() << " microseconds" << std::endl;

	return inliersResult;

}

std::unordered_set<int> Ransac2D_mul_thrd(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	// Time segmentation process
	auto startTime = std::chrono::steady_clock::now();
	
	std::unordered_set<int> inliersResult;
	srand(time(NULL));
	
	while (maxIterations --)
	{
		// Randomly pick two points
		std::unordered_set<int> inliers;
		while (inliers.size()< 2)
		    inliers.insert(rand()%(cloud->points.size()));

		float x1, y1, x2, y2;

		auto itr = inliers.begin();
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;

		float a = (y1-y2);
		float b = (x2-x1);
		float c = (x1*y2 - x2*y1);

		float new_distanceTol = distanceTol * sqrt(a*a + b*b);

		std::mutex m;
		parallel_for (cloud->points.size(), [&] (int start, int end)
		{
			for (size_t i = 0; i < cloud->points.size(); ++i)
			{
				if (inliers.count(i)>0) continue;
				
				pcl::PointXYZ point = cloud->points[i];
				float x3 = point.x;
				float y3 = point.y;
				
				float d = fabs(a*x3 + b*y3 + c);
				if (d < new_distanceTol) 
				{
					std::lock_guard<std::mutex> lk(m);
					inliers.insert(i);
				}
            }
        });

		if (inliers.size() > inliersResult.size())
		{
			inliersResult = inliers;
		}

		if (inliers.size() > cloud->points.size() * 0.4) break;

	}

	auto endTime = std::chrono::steady_clock::now();
	auto elapseTime = std::chrono::duration_cast<std::chrono::microseconds> (endTime - startTime);
	std::cout << "Ransac took " << elapseTime.count() << " microseconds" << std::endl;

	return inliersResult;

}

std::unordered_set<int> Ransac3D(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	// Time segmentation process
	auto startTime = std::chrono::steady_clock::now();
	
	std::unordered_set<int> inliersResult;
	srand(time(NULL));

	// For max iterations 

	// Randomly sample subset and fit line

	// Measure distance between every point and fitted line
	// If distance is smaller than threshold count it as inlier

	// Return indicies of inliers from fitted line with most inliers
	while (maxIterations --)
	{
		
		// Randomly pick three points
		std::unordered_set<int> inliers;
		while (inliers.size()< 3)
		    inliers.insert(rand()%(cloud->points.size()));

		auto startTime = std::chrono::steady_clock::now();    

		float x1, y1, z1, 
		      x2, y2, z2, 
			  x3, y3, z3;

		auto itr = inliers.begin();
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		z1 = cloud->points[*itr].z;
		
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;
		z2 = cloud->points[*itr].z;

		itr++;
		x3 = cloud->points[*itr].x;
		y3 = cloud->points[*itr].y;
		z3 = cloud->points[*itr].z;

		float A = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1);
		float B = (z2-z1)*(x3-x1) - (x2-x1)*(z3-z1);
		float C = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1);

		if (A == 0 && B == 0 && C == 0) continue;

		float D = -(A*x1 + B*y1 + C*z1);

		float new_distanceTol = distanceTol * sqrt(A*A + B*B + C*C);

		int inliersTol = cloud->points.size() * 0.9;

		for(int index = 0; index < cloud->points.size(); index++)
		{
			if (inliers.count(index)>0) continue;

			pcl::PointXYZ point = cloud->points[index];
			float x3 = point.x;
			float y3 = point.y;
			float z3 = point.z;

			float d = fabs(A*x3 + B*y3 + C*z3 + D);
			if (d < new_distanceTol) inliers.insert(index);
			
		}

		if (inliers.size() > inliersResult.size()) { inliersResult = inliers;}

		if (inliers.size() > inliersTol) 
		{
			std::cout << "Early stopped at: " << maxIterations << std::endl;
			break;
		}

	}

	auto endTime = std::chrono::steady_clock::now();
	auto elapseTime = std::chrono::duration_cast<std::chrono::microseconds> (endTime - startTime);
	std::cout << "Ransac took " << elapseTime.count() << " microseconds" << std::endl;

	return inliersResult;

}

std::unordered_set<int> Ransac3D_SSE(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	// Time segmentation process
	auto startTime = std::chrono::steady_clock::now();
	
	std::unordered_set<int> inliersResult;
	srand(time(NULL));

	// For max iterations 

	// Randomly sample subset and fit line

	// Measure distance between every point and fitted line
	// If distance is smaller than threshold count it as inlier

	// Return indicies of inliers from fitted line with most inliers
	while (maxIterations --)
	{
		
		// Randomly pick two points
		std::unordered_set<int> inliers;
		while (inliers.size()< 3)
		    inliers.insert(rand()%(cloud->points.size()));  

		auto startTime = std::chrono::steady_clock::now();

		fvec4 x1, y1, z1, 
		      x2, y2, z2, 
			  x3, y3, z3;

		auto itr = inliers.begin();
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		z1 = cloud->points[*itr].z;
		
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;
		z2 = cloud->points[*itr].z;

		itr++;
		x3 = cloud->points[*itr].x;
		y3 = cloud->points[*itr].y;
		z3 = cloud->points[*itr].z;

		fvec4 A = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1);
		fvec4 B = (z2-z1)*(x3-x1) - (x2-x1)*(z3-z1);
		fvec4 C = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1);

		if (A[0] == 0 && B[0] == 0 && C[0] == 0) continue;

		fvec4 D = -(A*x1 + B*y1 + C*z1);

		fvec4 new_distanceTol = fvec4(distanceTol) * f_sqrt(A*A + B*B + C*C);

		int inliersTol = cloud->points.size() * 0.9;

	    int index;

	    for(index = 0; index < cloud->points.size() - 3; index+= 4)
		{
			//if (inliers.count(index)>0) continue;
			fvec4 x4 = fvec4 (cloud->points[index].x, 
			                  cloud->points[index+1].x, 
							  cloud->points[index+2].x, 
							  cloud->points[index+3].x);
			fvec4 y4 = fvec4 (cloud->points[index].y, 
			                  cloud->points[index+1].y, 
							  cloud->points[index+2].y, 
							  cloud->points[index+3].y);
			fvec4 z4 = fvec4 (cloud->points[index].z, 
			                  cloud->points[index+1].z, 
							  cloud->points[index+2].z, 
							  cloud->points[index+3].z);;

			fvec4 d = (A*x4 + B*y4 + C*z4 + D);

			if (d[0] < new_distanceTol[0] && inliers.count(index) == 0 ) inliers.insert(index);
			if (d[1] < new_distanceTol[0] && inliers.count(index+1) == 0) inliers.insert(index+1);
			if (d[2] < new_distanceTol[0] && inliers.count(index+2) == 0) inliers.insert(index+2);
			if (d[3] < new_distanceTol[0] && inliers.count(index+3) == 0) inliers.insert(index+3);

		}

		for( ; index < cloud->points.size(); index++)
		{
			if (inliers.count(index)>0) continue;

			pcl::PointXYZ point = cloud->points[index];
			fvec4 x3 = point.x;
			fvec4 y3 = point.y;
			fvec4 z3 = point.z;

			fvec4 d = (A*x3 + B*y3 + C*z3 + D);

			if (d[0] < new_distanceTol[0]) inliers.insert(index);
			
		}

		if (inliers.size() > inliersResult.size())
		{
			inliersResult = inliers;
		}

		if (inliers.size() > inliersTol) 
		{
			std::cout << "Early stopped at: " << maxIterations << std::endl;
			break;
		}

	}

	auto endTime = std::chrono::steady_clock::now();
	auto elapseTime = std::chrono::duration_cast<std::chrono::microseconds> (endTime - startTime);
	std::cout << "Ransac_sse took " << elapseTime.count() << " microseconds" << std::endl;

	return inliersResult;

}

int main ()
{

	// Create viewer
	pcl::visualization::PCLVisualizer::Ptr viewer = initScene();

	// Create data
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData2D();
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData3D();

	// TODO: Change the max iteration and distance tolerance arguments for Ransac function
	//std::unordered_set<int> inliers = Ransac3D_SSE(cloud, 20, 0.3);
	std::unordered_set<int> inliers = Ransac3D(cloud, 20, 0.3);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudInliers(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());

	for(int index = 0; index < cloud->points.size(); index++)
	{
		pcl::PointXYZ point = cloud->points[index];
		if (inliers.count(index))
			cloudInliers->points.emplace_back(point);
		else
			cloudOutliers->points.emplace_back(point);
	}

	
	// Render 2D point cloud with inliers and outliers
	if(inliers.size())
	{
		renderPointCloud(viewer,cloudInliers,"inliers",Color(0,1,0));
  		renderPointCloud(viewer,cloudOutliers,"outliers",Color(1,0,0));
	}
  	else
  	{
  		renderPointCloud(viewer,cloud,"data");
  	}
	
  	while (!viewer->wasStopped ())
  	{
  	  viewer->spinOnce ();
  	}

	/* Test SSE instruction

	float x = 2.0;
	float sqrt_x, sse_rsqrt_x, sse_sqrt_x, SSESqrt_Recip_Times_x;
    
	auto startTime = std::chrono::steady_clock::now();

	for (int i = 0; i < 100000; i++) {sqrt_x = SqrtFunction(x);}

    auto endTime = std::chrono::steady_clock::now();
    auto elapseTime = std::chrono::duration_cast<std::chrono::nanoseconds> (endTime - startTime);
	std::cout << "sqrt took " << sqrt_x <<":"<< elapseTime.count() << " nanoseconds" << std::endl;


    startTime = std::chrono::steady_clock::now();
	//for (int i = 0; i < 100000; i++) {sse_sqrt(&sse_sqrt_x,&x);}
	for (int i = 0; i < 100000; i++) { store(&sse_sqrt_x, f_sqrt(fvec4(x)));}

	endTime = std::chrono::steady_clock::now();
    elapseTime = std::chrono::duration_cast<std::chrono::nanoseconds> (endTime - startTime);
	std::cout << "sse_sqrt took " << sse_sqrt_x <<":"<< elapseTime.count() << " nanoseconds" << std::endl;

    startTime = std::chrono::steady_clock::now();
	//for (int i = 0; i < 100000; i++) {sse_rsqrt(&sse_rsqrt_x,&x);}
	for (int i = 0; i < 100000; i++) { store(&sse_rsqrt_x, f_RTsqrt(fvec4(x)));}


	endTime = std::chrono::steady_clock::now();
    elapseTime = std::chrono::duration_cast<std::chrono::nanoseconds> (endTime - startTime);
	std::cout << "f_RTsqrt took " << sse_rsqrt_x <<":"<< elapseTime.count() << " nanoseconds" << std::endl;
    */

  	
}
