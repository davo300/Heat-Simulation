// COMP-4400 - Parallel Programming A
// Matt Davies - ID: 110121897


#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <string>


/*
tempToBGR
It maps temperatures to colors for visualization:
Cold -> blueish
Medium -> greenish
Hot -> reddish
*/

static inline cv::Vec3b tempToBGR(double t, double tmin, double tmid, double tmax) {
    if (t <= tmid) {
        double f = (t - tmin) / (tmid - tmin);
        f = std::min(1.0, std::max(0.0, f));
        return cv::Vec3b((uchar)std::lround(255*f), 255, (uchar)std::lround(255*f));
    } else {
        double g = (t - tmid) / (tmax - tmid);
        g = std::min(1.0, std::max(0.0, g));
        return cv::Vec3b((uchar)std::lround(255*(1-g)), (uchar)std::lround(255*(1-g)), 255);
    }
}


static void set_bounds(cv::Mat &grid, cv::Mat &fixedMask, const std::string &side,
                           double lenFrac=0.30, int thickness=4) {
    int H = grid.rows, W = grid.cols;
    const double wall=20.0, heater=100.0;
    fixedMask.setTo(0);
    // set values to the matrix
    grid.row(0).setTo(wall);       
    grid.row(H-1).setTo(wall);     
    grid.col(0).setTo(wall);       
    grid.col(W-1).setTo(wall);

    fixedMask.row(0).setTo(255);
    fixedMask.row(H-1).setTo(255);
    fixedMask.col(0).setTo(255);
    fixedMask.col(W-1).setTo(255);

    // double check the lenFrac and thickness
    lenFrac = std::clamp(lenFrac, 0.05, 1.0);
    thickness = std::max(1, thickness);

    if (side=="top") {      // the default heater location
        int L = std::max(1, (int)std::lround(W*lenFrac));
        int x0 = (W-L)/2, x1 = x0+L-1;
        int t = std::min(thickness, H);
        grid(cv::Range(0,t), cv::Range(x0,x1+1)).setTo(heater);
        fixedMask(cv::Range(0,t), cv::Range(x0,x1+1)).setTo(255);
    } else if (side=="bottom") {
        int L = std::max(1, (int)std::lround(W*lenFrac));
        int x0 = (W-L)/2, x1 = x0+L-1;
        int t = std::min(thickness, H);
        grid(cv::Range(H-t,H), cv::Range(x0,x1+1)).setTo(heater);
        fixedMask(cv::Range(H-t,H), cv::Range(x0,x1+1)).setTo(255);
    } else if (side=="left") {
        int L = std::max(1, (int)std::lround(H*lenFrac));
        int y0 = (H-L)/2, y1 = y0+L-1;
        int t = std::min(thickness, W);
        grid(cv::Range(y0,y1+1), cv::Range(0,t)).setTo(heater);
        fixedMask(cv::Range(y0,y1+1), cv::Range(0,t)).setTo(255);
    } else if (side=="right") {
        int L = std::max(1, (int)std::lround(H*lenFrac));
        int y0 = (H-L)/2, y1 = y0+L-1;
        int t = std::min(thickness, W);
        grid(cv::Range(y0,y1+1), cv::Range(W-t,W)).setTo(heater);
        fixedMask(cv::Range(y0,y1+1), cv::Range(W-t,W)).setTo(255);
    } else { // default top
        int L = std::max(1, (int)std::lround(W*lenFrac));
        int x0 = (W-L)/2, x1 = x0+L-1;
        int t = std::min(thickness, H);
        grid(cv::Range(0,t), cv::Range(x0,x1+1)).setTo(heater);
        fixedMask(cv::Range(0,t), cv::Range(x0,x1+1)).setTo(255);
    }
}

int main(int argc, char** argv) {
    // --- Parse input arguments ---
    if (argc < 3) {         
        std::cerr << "usage: " << argv[0]
                  << " WIDTH HEIGHT [epsilon=1e-2] [heater=top|bottom|left|right]\n";
        return 1;
    }
    const int W = std::stoi(argv[1]), H = std::stoi(argv[2]);
    double eps;
    if (argc >= 4)  // convergence tolerance
        eps = std::stod(argv[3]); // 1e-2 == 1×10^(−2)=0.01
    else 
        eps = 1e-2;     // 1e-2 == 1×10^(−2)=0.01
    std::string heaterSide; 
    if (argc >= 5)  // heater location
        heaterSide = std::string(argv[4]);  // use 5th command-line arg
    else
        heaterSide = "top";     // default arg
    if (W < 3 || H < 3) { std::cerr << "WIDTH and HEIGHT must be >= 3\n"; return 1; }

    // --- Temperature ranges ---
    const double Tinit = -20.0, Tmin = -20.0, Tmax = 100.0, Tmid = 0.5*(Tmin + Tmax);

    // --- Initialize grids ---
    cv::Mat curr(H, W, CV_64F, cv::Scalar(Tinit));  // current temperature grid
    cv::Mat next(H, W, CV_64F, cv::Scalar(Tinit));  // next temperature grid
    cv::Mat fixed(H, W, CV_8U, cv::Scalar(0));      // mask of fixed cells
    set_bounds(curr, fixed, heaterSide, 0.30, 4); // set walls and heater
    curr.copyTo(next);

    cv::Mat rgb(H, W, CV_8UC3);                     // visualization image
    cv::namedWindow("heat distribution", cv::WINDOW_AUTOSIZE);

    size_t iter = 0;

    // --- Main iteration loop ---
    for (;;) {      // infinte loop
        double maxDiff = 0.0; // track maximum temperature change

        // --- Compute new temperatures (Jacobi method) ---
        for (int i = 1; i < H-1; ++i) {
            const double* up = curr.ptr<double>(i-1);
            const double* mid = curr.ptr<double>(i);
            const double* dn = curr.ptr<double>(i+1);
            double* nrow = next.ptr<double>(i);
            const uchar* mask = fixed.ptr<uchar>(i);

            for (int j = 1; j < W-1; ++j) {
                if (mask[j]) { nrow[j] = mid[j]; continue; } // skip fixed cells
                double newT = 0.25*(mid[j-1] + mid[j+1] + up[j] + dn[j]); // average neighbors
                nrow[j] = newT;
                maxDiff = std::max(maxDiff, std::abs(newT - mid[j]));
            }
        }

        // --- Re-apply fixed temperatures (walls/heater) ---
        for (int i = 0; i < H; ++i) {
            const uchar* mask = fixed.ptr<uchar>(i);
            const double* c = curr.ptr<double>(i);
            double* n = next.ptr<double>(i);
            for (int j = 0; j < W; ++j) 
                if (mask[j]) n[j] = c[j];
        }

        std::swap(curr, next); // update current grid

        // --- Colorize for display ---
        for (int i = 0; i < H; ++i) {
            const double* row = curr.ptr<double>(i);
            cv::Vec3b* out = rgb.ptr<cv::Vec3b>(i);
            for (int j = 0; j < W; ++j) 
                out[j] = tempToBGR(row[j], Tmin, Tmid, Tmax);   // important color function call.
        }

        // --- Draw heater overlay for visualization ---
        if (heaterSide == "top" || heaterSide == "bottom") {
            int L = std::lround(W * 0.30);
            int x0 = (W-L)/2;
            int t = 4;
            cv::Rect r;
            if (heaterSide == "top")
                r = cv::Rect(x0, 0, L, std::min(t, H));
            else
                r = cv::Rect(x0, std::max(0, H - t), L, std::min(t, H));
            cv::rectangle(rgb,r,cv::Scalar(0,0,255),cv::FILLED); // red heater
        } else {
            int L = std::lround(H * 0.30);
            int y0 = (H-L)/2;
            int t = 4;
            cv::Rect r;
            if (heaterSide == "left") 
                r = cv::Rect(0, y0, std::min(t, W), L);
            else 
                r = cv::Rect(std::max(0, W - t), y0, std::min(t, W), L);
            cv::rectangle(rgb,r,cv::Scalar(0,0,255),cv::FILLED);
        }

        // --- Display ---
        cv::Mat vis = rgb.clone();
        cv::imshow("heat distribution", vis);

        int key = cv::waitKey(1); // wait 1ms for keypress
        if (key == 'q' || key == 27) 
            break; // quit on 'q' or ESC
        ++iter;
        if (maxDiff < eps) 
            break; // stop when converged
    }

    cv::waitKey(0); // wait for user to close
    return 0;
}