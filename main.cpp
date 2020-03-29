#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace std;
using namespace cv;

Mat fruta, bird, cano;

void detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale);
void drawBird(Mat frame, int ry);
void drawScenery(Mat frame, int incrementCenario, int frames);

string cascadeName;
string nestedCascadeName;

/**
 * @brief Draws a transparent image over a frame Mat.
 * 
 * @param frame the frame where the transparent image will be drawn
 * @param transp the Mat image with transparency, read from a PNG image, with the IMREAD_UNCHANGED flag
 * @param xPos x position of the frame image where the image will start.
 * @param yPos y position of the frame image where the image will start.
**/

// Não to usando
void drawTransparency(Mat frame, Mat transp, int xPos, int yPos){
  Mat mask;
  vector<Mat> layers;
    
  split(transp, layers); // seperate channels
  Mat rgb[3] = { layers[0], layers[1], layers[2] };
  mask = layers[3]; // png's alpha channel used as mask
  merge(rgb, 3, transp);  // put together the RGB channels, now transp insn't transparent 
  transp.copyTo(frame.rowRange(yPos, yPos + transp.rows).colRange(xPos, xPos + transp.cols));
}

void drawTransparency2(Mat frame, Mat transp, int xPos, int yPos){
  Mat mask;
  vector<Mat> layers;
  
  split(transp, layers); // seperate channels
  Mat rgb[3] = { layers[0],layers[1],layers[2] };
  mask = layers[3]; // png's alpha channel used as mask
  merge(rgb, 3, transp);  // put together the RGB channels, now transp insn't transparent 
  Mat roi1 = frame(Rect(xPos, yPos, transp.cols, transp.rows));
  Mat roi2 = roi1.clone();
  transp.copyTo(roi2.rowRange(0, transp.rows).colRange(0, transp.cols), mask);
  //printf("%p, %p\n", roi1.data, roi2.data);
  double alpha = 1; // Transparencia da imagem
  addWeighted(roi2, alpha, roi1, 1 - alpha, 0.0, roi1);
}

int main(int argc, const char** argv){
  VideoCapture capture;
  Mat frame, image;
  string inputName;
  CascadeClassifier cascade, nestedCascade;
  const double scale = 1;

  
  /**fruta = cv::imread("laranja.png", IMREAD_UNCHANGED);
    if(fruta.empty()){
      printf("Error opening file laranja.png\n");
    }
  **/

  // Abre a imagem do personagem
  bird = cv::imread("bird.png");
  if(bird.empty()){
    cout << "Error opening file bird.png" << endl;
  }

  // Abre a imagem do cano do cenario
  cano = cv::imread("cano.png");
  if(cano.empty()){
    cout << "Error opening file cano.png" << endl;
  }

  string folder = "/home/rosivaldo/Downloads/opencv-4.1.2/data/haarcascades/";
  cascadeName = folder + "haarcascade_frontalface_alt.xml";
  nestedCascadeName = folder + "haarcascade_eye_tree_eyeglasses.xml";
  inputName = "/dev/video0";
  //inputName = inputName = "video2019.2.mp4";

  if(!nestedCascade.load(samples::findFileOrKeep(nestedCascadeName))){
    cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
  }

  if(!cascade.load(samples::findFile(cascadeName))){
    cerr << "ERROR: Could not load classifier cascade" << endl;
    return -1;
  }

  if(!capture.open(inputName)){
    cout << "Capture from camera #" << inputName << " didn't work" << endl;
    return 1;
  }

  // Se conseguiu abrir a câmera entra nesse if
  if(capture.isOpened()){
    cout << "Video capturing has been started ..." << endl;

    for(;;){
      capture >> frame;

      if(frame.empty()){
        break;
      }

      //Mat frame1 = frame.clone();
      detectAndDraw(frame, cascade, nestedCascade, scale);

      char c = (char)waitKey(10);
      if(c == 27 || c == 'q' || c == 'Q'){
        break;
      }
    }
  }

  return 0;
}

void detectAndDraw(Mat& frame, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale){
  static int frames = 0; // Conta o número de frames
  double t = 0; // Conta o tempo de execução do programa
  vector<Rect> faces; //faces2; // Vetor de faces

  const static Scalar colors[] = { // Determina as cores dos retangulos e circulos da face
    Scalar(255,0,0),
    Scalar(255,128,0),
    Scalar(255,255,0),
    Scalar(0,255,0),
    Scalar(0,128,255),
    Scalar(0,255,255),
    Scalar(0,0,255),
    Scalar(255,0,255)
  };

  Mat gray, smallImg;

  cvtColor(frame, gray, COLOR_BGR2GRAY);
  double fx = 1 / scale;
  resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
  equalizeHist(smallImg, smallImg);

  t = (double)getTickCount();

  // |CASCADE_FIND_BIGGEST_OBJECT |CASCADE_DO_ROUGH_SEARCH
  cascade.detectMultiScale(smallImg, faces, 1.2, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30));

  frames++; // Incrementa o frame

  // A cada 30 frames toca um som
  /*if(frames % 30 == 0){
    system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
  }*/

  t = (double)getTickCount() - t;
  //printf("detection time = %g ms\n", t*1000/getTickFrequency());
  
  // Se entrar aqui é pq detectou a face
  for(size_t i = 0; i < faces.size(); i++){
    Rect r = faces[i];
    printf("[%3d, %3d]\n", r.x, r.y);
    Mat smallImgROI;
    vector<Rect> nestedObjects;
    Point center;
    Scalar color = colors[i%8];
    int radius;
    //int move = 0;

    rectangle(frame, Point(cvRound(r.x*scale), cvRound(r.y*scale)), Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0);

    drawBird(frame, r.y); // Desenha o personagem na tela
    drawScenery(frame, 100, frames); // Desenha os canos na tela
    
    if(nestedCascade.empty()){
      continue;
    }

    smallImgROI = smallImg(r);

    //|CASCADE_FIND_BIGGEST_OBJECT |CASCADE_DO_ROUGH_SEARCH |CASCADE_DO_CANNY_PRUNING
    nestedCascade.detectMultiScale(smallImgROI, nestedObjects, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30));
    
    // Se entrar aqui é pq detectou o olho
    for(size_t j = 0; j < nestedObjects.size(); j++){
      Rect nr = nestedObjects[j];
      center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
      center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
      radius = cvRound((nr.width + nr.height)*0.25*scale);
      circle(frame, center, radius, color, 3, 8, 0);
    }
  }

  if(!fruta.empty()){
    drawTransparency2(frame, fruta, 100, 100);
  }

  cv::putText(frame, // target image
    "Meu texto de teste", // text
    cv::Point(50, 50), // top-left position
    cv::FONT_HERSHEY_DUPLEX,
    1.0,
    CV_RGB(255, 0, 0), // font color
    2
  );

  imshow("Flappy Bird", frame);
}

// Função para desenhar e movimentar o personagem
void drawBird(Mat frame, int ry){
  static int x = 280; // Inicio da imagem
  static int y = 240; // Inicio da imagem
  static int ryFun = ry; // Recebe a posição inicial da coordenada y
  
  if(ry >= ryFun && ry <= ryFun){
    // colRange() -> x / rowRange() -> y
    x += 1;
    bird.copyTo(frame.colRange(x, x + bird.cols).rowRange(y, y + bird.rows));
  }
  
  if(ry > ryFun){
    x += 1;
    y += 10;
    bird.copyTo(frame.colRange(x, x + bird.cols).rowRange(y, y + bird.rows));
  }

  if(ry < ryFun){
    x -= 1;
    y -= 10;
    bird.copyTo(frame.colRange(x, x + bird.cols).rowRange(y, y + bird.rows)); 
  }
}

void drawScenery(Mat frame, int incrementCenario, int frames){
  const int y = 0; // Max width (x) = 550 Max height (y) = 250
  static int x = 0;
  static int x2 = 100;
  static int x3 = 200;
  static int x4 = 300;
  //if(frames % 30 == 0){
    
    cano.copyTo(frame.colRange(x, x + cano.cols).rowRange(y, y + cano.rows));
    cano.copyTo(frame.colRange(x2, x2 + cano.cols).rowRange(50, 50 + cano.rows));
    cano.copyTo(frame.colRange(x3, x3 + cano.cols).rowRange(20, 20 + cano.rows));
    cano.copyTo(frame.colRange(x4, x4 + cano.cols).rowRange(20, 20 + cano.rows));
    /*cano.copyTo(frame.colRange(x, x + cano.cols).rowRange(y, y + cano.rows));
    cano.copyTo(frame.colRange(x, x + cano.cols).rowRange(y, y + cano.rows));
    cano.copyTo(frame.colRange(x, x + cano.cols).rowRange(y, y + cano.rows));*/
    //cano.copyTo(frame.colRange(x, x + cano.cols).rowRange(y, y + cano.rows));
    
    /*if(x  x2){
      cout << "colidiu" << endl;
    }*/
    //cano.copyTo(frame.colRange(100, 100 + cano.cols).rowRange(50, 50 + cano.rows));
    //cano.copyTo(frame.colRange(200, 200 + cano.cols).rowRange(y, y + cano.rows));
    //cano.copyTo(frame.colRange(300, 300 + cano.cols).rowRange(50, 50 + cano.rows));
  //}else{
    //cano.copyTo(frame.colRange(50, 50 + cano.cols).rowRange(30, 30 + cano.rows));
    //cano.copyTo(frame.colRange(150, 150 + cano.cols).rowRange(y, y + cano.rows));
    //cano.copyTo(frame.colRange(250, 250 + cano.cols).rowRange(30, 30 + cano.rows));
    //cano.copyTo(frame.colRange(350, 350 + cano.cols).rowRange(y, y + cano.rows));
  //}
  if(frames % 5 == 0){
    x += 50;
    x2 += 50;
    x3 += 50;
    x4 += 50;
  }
  /*if(x >= 550){
      x = 0;
    }*/
    /*if(x2 >= 550){
      x2 = 100;
    }*/
    if(x4 >= 550){
      x = 0;
      x2 = 100;
      x3 = 200;
      x4 = 300;
    }
}
