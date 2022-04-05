#include <iostream>

#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include "opencv2/opencv.hpp"
#include <string>

#define ERROR_CHECK_STATUS(status) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
}

#define ERROR_CHECK_OBJECT(obj) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
}

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[]) {
	size_t len = strlen(string);
	if (len > 0) {
		printf("%s", string);
		if (string[len - 1] != '\n')
			printf("\n");
		fflush(stdout);
	}
}

using namespace cv;
using namespace std;

int main() {
//	Setup vx context and graph
	vx_context context = vxCreateContext();
	ERROR_CHECK_OBJECT(context);
	vxRegisterLogCallback(context, log_callback, vx_false_e);

	vx_graph graph = vxCreateGraph(context);
	ERROR_CHECK_OBJECT(graph);

//	OpenCV read image file
//  TODO: use imread to get image dimension
	Mat input_yds = imread("../yonge_dundas_square.jpg");
	Mat input_uoft = imread("../uoft_soldiers_tower_dark.png");

//	Set image width and height
	int width_yds = 480, height_yds = 360;
	int width_uoft = 480, height_uoft = 360;
	int width_target = 480, height_target = 360;

//	Initialize vx images
	vx_image input_yds_image = vxCreateImage(context, width_yds, height_yds, VX_DF_IMAGE_RGB);
	vx_image input_uoft_image = vxCreateImage(context, width_uoft, height_uoft, VX_DF_IMAGE_RGB);
	vx_image histE_image = vxCreateImage(context, width_yds, height_yds, VX_DF_IMAGE_U8);
	vx_image bilinear_interpolated_image = vxCreateImage(context, width_target, height_target, VX_DF_IMAGE_U8);
	ERROR_CHECK_OBJECT(input_yds_image);
	ERROR_CHECK_OBJECT(input_uoft_image);
	ERROR_CHECK_OBJECT(histE_image);
	ERROR_CHECK_OBJECT(bilinear_interpolated_image);

//	VX matrix to store Homography matrix (3 by 3)
vx_matrix H = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);

// Nodes used in graph
	vx_node nodes[] =
		{
			vxEqualizeHistNode(graph, input_yds_image, histE_image),
			vxWarpPerspectiveNode(graph, input_uoft_image, H, VX_INTERPOLATION_BILINEAR, bilinear_interpolated_image)
		};

//	Check node status
	for( vx_size i = 0; i < sizeof( nodes ) / sizeof( nodes[0] ); i++ )
	{
		ERROR_CHECK_OBJECT( nodes[i] );
		ERROR_CHECK_STATUS( vxReleaseNode( &nodes[i] ) );
	}

//	Verify graph
	ERROR_CHECK_STATUS( vxVerifyGraph( graph ) );

	return 0;
}
