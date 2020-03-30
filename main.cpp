#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace std;
using namespace cv;

string caminhos[] = {
  "cano1.png",
  "cano2.png",
  "cano3.png",
  "cano4.png",
  "cano_baixo01.png",
  "cano_baixo02.png",
  "cano_baixo03.png",
  "cano_baixo04.png"
};

Mat bird, fundo;
Mat canos[8];

void detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale);
void drawBird(Mat frame, int ry, int frames);
void drawScenery(Mat frame, int frames);
void detectCollision(int yBird, int velocidade);

string cascadeName;
string nestedCascadeName;
Rect r;
static int score = 0; // Pontos do jogo
string scoreString;
static int y = 240; // yBird
static int x = 0; // xBird

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
  int tamanho;
  
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

  tamanho = sizeof(canos)/sizeof(canos[0]);

  // Abre as imagem do cenario
  for(int i = 0; i < tamanho; i++){
    canos[i] = cv::imread(caminhos[i]);
    if(canos[i].empty()){
      cout << "Erro opening file " << caminhos[i] << ".png" << endl;
    }
  }
  
  // Abre a imagem do cano do cenario
  fundo = cv::imread("nuvem.png");
  if(fundo.empty()){
    cout << "Error opening file cano.png" << endl;
  }

  // Abre a imagem do cano2 do cenario
  //canos[1] = cv::imread("cano2.png");
  //if(cano2.empty()){
  //  cout << "Error opening file cano2.png" << endl;
  //}

  // Abre a imagem do canoLongo do cenario
  //canos[2] = cv::imread("cano_longo.png");
  //if(canoLongo.empty()){
   // cout << "Error opening file cano_longo.png" << endl;
 // }

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
  //if(frames % 30 == 0){
    //system("mplayer /home/rosivaldo/Documentos/Faculdade/2019.2/Programacao1/FlappyBird/som.mp3 &");
  //}
  //string q;
  //cin >> q;
  //system("mplayer [q] [/home/rosivaldo/Documentos/Faculdade/2019.2/Programacao1/FlappyBird/som.mp3]");

  t = (double)getTickCount() - t;
  //printf("detection time = %g ms\n", t*1000/getTickFrequency());
  
  // Se entrar aqui é pq detectou a face
  for(size_t i = 0; i < faces.size(); i++){
    r = faces[i];
    //printf("[%3d, %3d]\n", r.x, r.y);
    Mat smallImgROI;
    vector<Rect> nestedObjects;
    Point center;
    Scalar color = colors[i%8];

    rectangle(frame, Point(cvRound(r.x*scale), cvRound(r.y*scale)), Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0);
    
    if(nestedCascade.empty()){
      continue;
    }

    smallImgROI = smallImg(r);

    //|CASCADE_FIND_BIGGEST_OBJECT |CASCADE_DO_ROUGH_SEARCH |CASCADE_DO_CANNY_PRUNING
    nestedCascade.detectMultiScale(smallImgROI, nestedObjects, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30));
    
  }

  drawScenery(frame, frames); // Desenha os canos na tela
  drawBird(frame, r.y, frames); // Desenha o personagem na tela

  //drawTransparency2(frame, fundo, 10, 10);
  fundo.copyTo(frame.colRange(0, 0 + fundo.cols).rowRange(0, 0 + fundo.rows));

  scoreString = to_string(score);

  cv::putText(frame, // target image
    "SCORE", // text
    cv::Point(255, 25), // top-left position
    cv::FONT_HERSHEY_DUPLEX,
    1.0,
    CV_RGB(0, 0, 0), // font color
    2
  );

  cv::putText(frame, // target image
    scoreString, // text
    cv::Point(290, 60), // top-left position
    cv::FONT_HERSHEY_DUPLEX,
    1.0,
    CV_RGB(0, 0, 0), // font color
    2
  );

  imshow("Flappy Bird", frame);
}

/**
 * 
 * FUNÇÕES DO PROGRAMA
 * 
**/

// Função para desenhar e movimentar o personagem
void drawBird(Mat frame, int ry, int frames){
  static int ryFun = ry; // Recebe a posição inicial da coordenada y
  static int velocidade = 2; // Velocidade inicial = 2 / Velocidade máxima = 7;

  if(ry >= ryFun && ry <= ryFun){
    bird.copyTo(frame.colRange(x, x + bird.cols).rowRange(y, y + bird.rows));
  }
  
  if(ry > ryFun){
    y += velocidade;
    x += velocidade;
    detectCollision(y, velocidade);
    bird.copyTo(frame.colRange(x, x + bird.cols).rowRange(y, y + bird.rows));
  }

  if(ry < ryFun){
    y -= velocidade;
    x += velocidade;
    detectCollision(y, velocidade);
    bird.copyTo(frame.colRange(x, x + bird.cols).rowRange(y, y + bird.rows));
  }

  // Verifica se chegou no limite da tela e reinicia
  if(x >= 525){
    x = 0;
    y = 240;
    velocidade += 1;
    if(velocidade > 7){
      velocidade = 2;
    }
  }
  
  // Conta os pontos
  if(frames % 15 == 0){
    score += 1;
  }
}

void drawScenery(Mat frame, int frames){
  const int y1 = 0; // Max width (x) = 550 Max height (y) = 250
  //const int y2 = 300;
  static int x1[] = {20, 100, 180, 250}; // Posições na coordenada x
  static int x2[] = {535, 470, 390, 320}; // Posições na coordenada x

  // Figura 3
  canos[2].copyTo(frame.colRange(x1[0], x1[0] + canos[2].cols).rowRange(y1, y1 + canos[2].rows));
  canos[2].copyTo(frame.colRange(x1[1], x1[1] + canos[2].cols).rowRange(y1, y1 + canos[2].rows));
  canos[2].copyTo(frame.colRange(x1[2], x1[2] + canos[2].cols).rowRange(y1, y1 + canos[2].rows));
  canos[2].copyTo(frame.colRange(x1[3], x1[3] + canos[2].cols).rowRange(y1, y1 + canos[2].rows));
  
  canos[2].copyTo(frame.colRange(x2[3], x2[3] + canos[2].cols).rowRange(y1, y1 + canos[2].rows));
  canos[2].copyTo(frame.colRange(x2[2], x2[2] + canos[2].cols).rowRange(y1, y1 + canos[2].rows));
  canos[2].copyTo(frame.colRange(x2[1], x2[1] + canos[2].cols).rowRange(y1, y1 + canos[2].rows));
  canos[2].copyTo(frame.colRange(x2[0], x2[0] + canos[2].cols).rowRange(y1, y1 + canos[2].rows));
  
  //Figura cano_baixo01
  canos[4].copyTo(frame.colRange(x1[0], x1[0] + canos[4].cols).rowRange(315, 315 + canos[4].rows));
  canos[4].copyTo(frame.colRange(x1[3], x1[3] + canos[4].cols).rowRange(315, 315 + canos[4].rows));
  canos[4].copyTo(frame.colRange(x1[1], x1[1] + canos[4].cols).rowRange(315, 315 + canos[4].rows));
  canos[4].copyTo(frame.colRange(x1[2], x1[2] + canos[4].cols).rowRange(315, 315 + canos[4].rows));
  
  canos[4].copyTo(frame.colRange(x2[3], x2[3] + canos[4].cols).rowRange(315, 315 + canos[4].rows));
  canos[4].copyTo(frame.colRange(x2[2], x2[2] + canos[4].cols).rowRange(315, 315 + canos[4].rows));
  canos[4].copyTo(frame.colRange(x2[1], x2[1] + canos[4].cols).rowRange(315, 315 + canos[4].rows));
  canos[4].copyTo(frame.colRange(x2[0], x2[0] + canos[4].cols).rowRange(315, 315 + canos[4].rows));
  //cout << "Cano baixo: " << canos[4].size().height << endl;

}

void detectCollision(int yBird, int velocidade){
  int a;
  
  if(velocidade == 2){
    if(yBird == 192){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
    if(yBird == 272){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
  }
  if(velocidade == 3){
    if(yBird == 198){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
    if(yBird == 273){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
  }
  if(velocidade == 4){
    if(yBird == 196){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
    if(yBird == 276){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
  }
  if(velocidade == 5){
    if(yBird == 195){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
    if(yBird == 275){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
  }
  if(velocidade == 6){
    if(yBird == 192){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
    if(yBird == 276){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
  }
  if(velocidade == 7){
    if(yBird == 191){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
    if(yBird == 282){
      system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");
      cin >> a;
    }
  }
}
