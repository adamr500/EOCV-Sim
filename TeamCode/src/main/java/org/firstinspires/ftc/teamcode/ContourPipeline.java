package org.firstinspires.ftc.teamcode;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;

public class ContourPipeline extends OpenCvPipeline {

    private double angle = 0;

    @Override
    public Mat processFrame(Mat input) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);

        // Define blue range
        Scalar lowerBlue = new Scalar(100, 150, 50);
        Scalar upperBlue = new Scalar(140, 255, 255);
        Mat mask = new Mat();
        Core.inRange(hsv, lowerBlue, upperBlue, mask);

        // Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Find the largest contour by area
        double maxArea = 0;
        MatOfPoint largestContour = null;
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                largestContour = contour;
            }
        }

        if (largestContour != null && !largestContour.empty()) {
            // Bounding rectangle
            Rect boundingRect = Imgproc.boundingRect(largestContour);
            double pixelHeight = boundingRect.height;

            // Distance estimation using bounding box height
            double realWorldLength = 0.09; // meters, real-world height of object
            double focalLengthPixels = 700; // Adjust this based on calibration
            double distanceMeters = (realWorldLength * focalLengthPixels) / pixelHeight;

            // Draw rectangle and center point
            Imgproc.rectangle(input, boundingRect, new Scalar(255, 255, 255), 2);
            Point center = new Point(
                boundingRect.x + boundingRect.width / 2.0,
                boundingRect.y + boundingRect.height / 2.0
            );

            // Angle calculation using the longest straight line in the contour (keep existing logic)
            double maxLength = 0;
            Point pt1 = null, pt2 = null;
            MatOfPoint2f contour2f = new MatOfPoint2f(largestContour.toArray());
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, 5, true);
            Point[] points = approx.toArray();
            for (int i = 0; i < points.length; i++) {
                Point p1 = points[i];
                Point p2_ = points[(i + 1) % points.length];
                double length = Math.hypot(p2_.x - p1.x, p2_.y - p1.y);
                if (length > maxLength) {
                    maxLength = length;
                    pt1 = p1;
                    pt2 = p2_;
                }
            }
            if (pt1 != null && pt2 != null) {
                Imgproc.line(input, pt1, pt2, new Scalar(255, 0, 0), 2);
                // Use distance-to-center logic for point ordering
                Point imageCenter = new Point(input.width() / 2.0, input.height() / 2.0);
                double dist1 = Math.hypot(pt1.x - imageCenter.x, pt1.y - imageCenter.y);
                double dist2 = Math.hypot(pt2.x - imageCenter.x, pt2.y - imageCenter.y);
                Point start = dist1 < dist2 ? pt1 : pt2;
                Point end = dist1 < dist2 ? pt2 : pt1;
                double dx = end.x - start.x;
                double dy = end.y - start.y;
                angle = Math.toDegrees(Math.atan2(dy, dx));
                if (angle < 0) {
                    angle += 360;
                }
                angle = angle % 180;
                // Display info near the bounding box center
                String infoText = String.format("Angle: %.2f\u00B0 | Dist: %.2fm", angle, distanceMeters);
                Imgproc.putText(input, infoText, new Point(center.x, center.y - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255), 2);
            }
        }

        return input;
    }

    public double getAngle() {
        return angle;
    }
}
