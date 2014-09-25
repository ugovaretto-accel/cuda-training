//OpenGL-CUDA interop
//Author: Ugo Varetto

//Note: GLUT is required in place of GLFW  when using a remote 
//client (vnc or vglconnect +vglrun) to avoid issues with the X11 keyboard
//extension./Requires GLM (OpenGL mathematics), to deal with the missing
//support for matrix stack in OpenGL >= 3.3

// nvcc -arch=sm_20 ../src/18_opengl-interop-glut.cu -DGL_GLEXT_PROTOTYPES \
// -I ../../../build/castor/local/glm/include \
// -L ../../../build/castor/local/glut/lib  -lGL -lglut

//with CUDA >= 6.5 and GLM >= 9.5
//nvcc -DCUDA_VERSION=6050 -DGL_GLEXT_PROTOTYPES -DGLM_COMPILER=0 -arch=sm_20 \
//../018_0_opengl-interop-glut.cu -lGL -L /usr/lib/x86_64-linux-gnu -lglut

//connection through vglconnect:
//1) log onto login node and ask for node: salloc -N1 -p special
//2) connect directly from workstation to node: vglconnect castor14
//3) run with vglrun: vglrun ./a,out ...

//#FANCY parameters 130 8 0.22 .001
//               or 130 4 0.1 0.2
//standard parameters e.g. 66 4 0.1 0.01

#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath> //isinf
#include <ctime>

#include <sys/time.h> //get time of day

#ifdef FANCY
#ifndef M_PI
#define M_PI 3.1415926535897932
#endif
#endif

#include <cuda_gl_interop.h>

#include <GL/freeglut.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "cuda_error_handler.h"

//use timer with resolution <= 10E6; GLUT_ELAPSED_TIME only has ms resolution
#include "Timer.h"

#define gle std::cout << "[GL] - " \
                      << __LINE__ << ' ' << glGetError() << std::endl;


//globals required by CUDA: since CUDA does not allow direct texture write, it
//is not possible to simply have a kernel that receives an input and output
//texture as easily done in OpenCL; cuda requires access to textures(input)
//and surfaces(output) bound to arrays and declared as global variables

//Mapping goes like this:
//OpenGL texture -->
//  CUDA Graphics resource -->
//    CUDA Array -->
//      CUDA Texture <float> [IN]
//      CUDA Surface <void> [OUT] !!!
///       use surf2Dwrite with x coordinate in bytes !!!


texture<float,2>  texIn; //read 
surface<void,2>  surfOut; //write

//------------------------------------------------------------------------------
GLuint create_program(const char* vertexSrc,
                      const char* fragmentSrc) {
    // Create the shaders
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    GLint res = GL_FALSE;
    int logsize = 0;
    // Compile Vertex Shader
    glShaderSource(vs, 1, &vertexSrc , NULL);
    glCompileShader(vs);

    // Check Vertex Shader
    glGetShaderiv(vs, GL_COMPILE_STATUS, &res);
    glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &logsize);
 
    if(logsize > 1) {
        std::cout << "Vertex shader:\n";
        std::vector<char> errmsg(logsize + 1, 0);
        glGetShaderInfoLog(vs, logsize, 0, &errmsg[0]);
        std::cout << &errmsg[0] << std::endl;
    }
    // Compile Fragment Shader
    glShaderSource(fs, 1, &fragmentSrc, 0);
    glCompileShader(fs);

    // Check Fragment Shader
    glGetShaderiv(fs, GL_COMPILE_STATUS, &res);
    glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &logsize);
    if(logsize > 1) {
        std::cout << "Fragment shader:\n";
        std::vector<char> errmsg(logsize + 1, 0);
        glGetShaderInfoLog(fs, logsize, 0, &errmsg[0]);
        std::cout << &errmsg[0] << std::endl;
    }

    // Link the program
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    // Check the program
    glGetProgramiv(program, GL_LINK_STATUS, &res);
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logsize);
    if(logsize > 1) {
        std::cout << "GLSL program:\n";
        std::vector<char> errmsg(logsize + 1, 0);
        glGetShaderInfoLog(program, logsize, 0, &errmsg[0]);
        std::cout << &errmsg[0] << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

//------------------------------------------------------------------------------
std::vector< float > create_2d_grid(int width, int height,
                                    int xOffset, int yOffset,
                                    float value) {
    std::vector< float > g(width * height);
#ifdef FANCY
    for(int y = 0; y != height; ++y) {
        for(int x = 0; x != width; ++x) {
            if(y <= yOffset )
                g[y * width + x] =
                    value * (sin(x * 16 * M_PI / width) 
                             + cos( y * 24 * M_PI / height));
            else if(y >= height - yOffset) 
                g[y * width + x] = 
                    value * (sin(x * 16 * M_PI / width) 
                             + cos((height - y) * 24 * M_PI / height));
            else if( x <= xOffset )
                g[y * width + x] = 
                    value * (sin(width * 16 * M_PI / width)  
                             + cos(y * 24 * M_PI / height));
            else if( x >= width - xOffset )
                g[y * width + x] = 
                   value * (sin((width)* 16 * M_PI / width) 
                             + cos(y * 24 * M_PI / height));
            else
                g[y * width + x] = float(0);
        }
    }
#else
    for(int y = 0; y != height; ++y) {
        for(int x = 0; x != width; ++x) {
            if(y < yOffset
               || x < xOffset
               || y >= height - yOffset
               || x >= width - xOffset)
                g[y * width + x] = value;
            else
                g[y * width + x] = float(0);     
    }
}
#endif    
    return g;
}


//------------------------------------------------------------------------------
void key_callback(unsigned char key, int xx, int yy) {
    switch(key) {
        case 27: // QUIT
            exit(EXIT_SUCCESS);
        default:
            ;    
    }
}    

//------------------------------------------------------------------------------
__device__ float laplacian(int x, int y) {
   const float v = tex2D(texIn, x, y);
   const float n = tex2D(texIn, x, y + 1);
   const float s = tex2D(texIn, x, y - 1);
   const float e = tex2D(texIn, x + 1, y);
   const float w = tex2D(texIn, x - 1, y);
   return (n + s + e + w - 4.0f * v);
}

__global__ void apply_stencil(float DIFFUSION_SPEED) {
   const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
   const int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
   const float v = tex2D(texIn, x, y);
   const float f = v + DIFFUSION_SPEED * laplacian(x, y);
   //WARNING: CUDA requires the x coordinate of various data
   //types to be in BYTES, NOT IN NUMBER OF ELEMENTS so
   //the x coordinate has to be multiplied by sizeof(element type)
   surf2Dwrite(f, surfOut,
               x * sizeof(float),
               y);
}


//------------------------------------------------------------------------------
//GLSL Shaders

//NOTE: it is important to keep the \n eol at the end of each line
//      to be able to easily match the line reported in the comiler
//      error to the location in the source code
const char fragmentShaderSrc[] =  //normalize value to map it to shades of gray
    "#version 330 core\n"
    "in vec2 UV;\n"
    "out vec3 color;\n"
    "uniform sampler2D cltexture;\n"
    "uniform float maxvalue;\n"
    "uniform float minvalue;\n"
    "void main() {\n"
    "  float c = texture2D(cltexture, UV).r;\n"
    "  c = smoothstep(minvalue, maxvalue, c);\n"
#ifdef FANCY    
    "  if(c < 0.1) color = vec3(6 * c, 0, 0);\n"
    "  else if(c >= 0.1 && c < 0.6) color = vec3(0.4, c, 0);\n"
    "  else if(c >= 0.6 && c < 0.8) color = vec3(0.5 * c, 0.4, c);\n"
    "  else color = vec3(0.3 * c, 0.2 * c, c);\n"
#else
    "  if(c < 0.5f) color=vec3(c, 0.1, 0.1);\n"
    "  else color = vec3(0.1, 0.1, c);\n"
#endif    
    "}";
const char vertexShaderSrc[] =
    "#version 330 core\n"
    "layout(location = 0) in vec2 pos;\n"
    "layout(location = 1) in vec2 tex;\n"
    "out vec2 UV;\n"
    "uniform mat4 MVP;\n"
    "void main() {\n"
    "  gl_Position = vec4(pos, 0.0f, 1.0f);\n"
    "  UV = tex;\n"
    "}";   

void f(){}
//------------------------------------------------------------------------------
bool IS_EVEN(int v) { return v % 2 == 0; }

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
//USER INPUT
    if(argc > 1 && argc < 4) {
      std::cout << "usage: " << argv[0]
                << "\n <size>\n"
                << " <workgroup size>\n"
                << "   (size - 2) DIV (workgroup size) = 0 (no remainder)\n"
                << " <diffusion speed>\n"
                << " [boundary value; default = 1]"
                << " negative values are rendered with shades of green"
                << std::endl; 
      exit(EXIT_FAILURE);          
    }
  
    const int STENCIL_SIZE = 3;
    const int SIZE = argc > 1 ? atoi(argv[1]) : 34;
    const int THREADS_PER_BLOCK = argc > 1 ? atoi(argv[2]) : 4;
    const int BLOCKS = (SIZE - 2 * (STENCIL_SIZE / 2)) / THREADS_PER_BLOCK;
    const float DIFFUSION_SPEED = argc > 1 ? atof(argv[3]) : 0.22;
    const float BOUNDARY_VALUE = argc > 4 ? atof(argv[4]) : 1.0f;
//GRAPHICS SETUP        

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    const int WWIDTH = 1024;
    const int WHEIGHT = 768;
    glutInitWindowSize(WWIDTH, WHEIGHT);
    glutKeyboardFunc(key_callback);
    glutDisplayFunc(f);
    glutIdleFunc(f);
    glutCreateWindow("CUDA - GL interop");   

//GEOMETRY AND CUDA-OPENGL MAPPING

    //geometry: textured quad; the texture color value is computed by CUDA
    float quad[] = {-1.0f,  1.0f,
                    -1.0f, -1.0f,
                     1.0f, -1.0f,
                     1.0f, -1.0f,
                     1.0f,  1.0f,
                    -1.0f,  1.0f};

    float texcoord[] = {0.0f, 1.0f,
                        0.0f, 0.0f,
                        1.0f, 0.0f,
                        1.0f, 0.0f,
                        1.0f, 1.0f,
                        0.0f, 1.0f};                 
    GLuint quadvbo;  
    glGenBuffers(1, &quadvbo);  
    glBindBuffer(GL_ARRAY_BUFFER, quadvbo);  
    glBufferData(GL_ARRAY_BUFFER, 6 * 2 * sizeof(float),
                 &quad[0], GL_STATIC_DRAW);   
    glBindBuffer(GL_ARRAY_BUFFER, 0);   

    GLuint texbo;  
    glGenBuffers(1, &texbo);
    glBindBuffer(GL_ARRAY_BUFFER, texbo);
    glBufferData(GL_ARRAY_BUFFER, 6 * 2 * sizeof(float),
                 &texcoord[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0); 

//CUDA-GL MAPPING
    //create textures mapped to CUDA buffers; initialize data in textures
    //directly

    std::vector< float > grid = create_2d_grid(SIZE, SIZE,
                                               STENCIL_SIZE / 2,
                                               STENCIL_SIZE / 2,
                                               BOUNDARY_VALUE);
    GLuint texEven;  
    glGenTextures(1, &texEven);

    glBindTexture(GL_TEXTURE_2D, texEven);
    
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_R32F, //IMPORTANT: required for unnormalized values;
                          //without this all the values in the texture
                          //are clamped to [0, 1];
                 SIZE,
                 SIZE,
                 0,
                 GL_RED,
                 GL_FLOAT,
                 &grid[0]);
    //optional
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    //required - use GL_NEAREST instead of GL_LINEAR to visualize
    //the actual discrete pixels
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
   
    glBindTexture(GL_TEXTURE_2D, 0);


    GLuint texOdd;  
    glGenTextures(1, &texOdd);
    glBindTexture(GL_TEXTURE_2D, texOdd);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_R32F, //IMPORTANT: required for unnormalized values;
                          //without this all the values in the texture
                          //are clamped to [0, 1];
                 SIZE,
                 SIZE,
                 0,
                 GL_RED,
                 GL_FLOAT,
                 &grid[0]);
    //optional
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    //required
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
    glBindTexture(GL_TEXTURE_2D, 0);


    //create CUDA buffers mapped to textures
    //cudaGraphicsRegisterFlagsSurfaceLoadStore  is required to be able
    //to read and write from / to textures with input textures and
    //output arrays
    cudaGraphicsResource* cudaBufferEven = 0;
    CUDA_CHECK(
        cudaGraphicsGLRegisterImage(
            &cudaBufferEven,
            texEven,
            GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsSurfaceLoadStore));
    cudaGraphicsResource* cudaBufferOdd = 0;
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
            &cudaBufferOdd,
            texOdd,
            GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsSurfaceLoadStore)); 
  

//OPENGL RENDERING SHADERS
    //create opengl rendering program

    GLuint glprogram = create_program(vertexShaderSrc, fragmentShaderSrc);
        
    //extract ids of shader variables
    GLuint mvpID = glGetUniformLocation(glprogram, "MVP");
    GLuint textureID = glGetUniformLocation(glprogram, "cltexture");
    GLuint maxValueID = glGetUniformLocation(glprogram, "maxvalue");
    GLuint minValueID = glGetUniformLocation(glprogram, "minvalue");

    //enable gl program
    glUseProgram(glprogram);

    //set texture id
    glUniform1i(textureID, 0); //always use texture 0

    //set min and max value; required to map it to shades of gray
    glUniform1f(maxValueID, BOUNDARY_VALUE);
    glUniform1f(minValueID, 0.0f);

//COMPUTE AND RENDER LOOP    
    int step = 0;
    GLuint tex = texEven;
    bool converged = false;
    std::cout << std::endl;
    double totalTime = 0;
    float prevError = 1;
    Timer timer;
    timer.Start();
   
    while(!converged) {     
        glutMainLoopEvent(); //<- due to a bug in freeglut this might indeed
                             //not work
//COMPUTE AND CHECK CONVERGENCE           
        glFinish(); //<-- ensure Open*G*L is done

        cudaArray* arrayIn = 0;
        cudaArray* arrayOut = 0;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaBufferEven));
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaBufferOdd));
           
        if(IS_EVEN(step)) {
            CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
                            &arrayIn,
                            cudaBufferEven,
                            0, 0));
            CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
                            &arrayOut,
                            cudaBufferOdd,
                            0, 0));
            tex = texOdd;
        } else {//even
            CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
                            &arrayOut,
                            cudaBufferEven,
                            0, 0));
            CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
                            &arrayIn,
                            cudaBufferOdd,
                            0, 0));
            tex = texEven;
        }

        CUDA_CHECK(cudaBindTextureToArray(texIn, arrayIn));
        CUDA_CHECK(cudaBindSurfaceToArray(surfOut, arrayOut));
        
        LAUNCH_CUDA_KERNEL(
            (apply_stencil<<<dim3(BLOCKS, BLOCKS, 1),
                             dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)
                          >>>(DIFFUSION_SPEED)));
        
#ifndef FANCY

        //CHECK FOR CONVERGENCE: extract element at grid center
        //and exit if |element value - boundary value| <= EPS    
        float centerOut = -BOUNDARY_VALUE;     
        
        CUDA_CHECK(cudaMemcpyFromArray(&centerOut,
                                       arrayOut,
                                       sizeof(float) * SIZE / 2,
                                       SIZE / 2,
                                       sizeof(float),
                                       cudaMemcpyDeviceToHost));

        const double elapsed = timer.Step() / 1E3; //ms --> s
        totalTime += elapsed;
       
        const float MAX_RELATIVE_ERROR = 0.01;//1%
        const float relative_error =
                fabs(centerOut - BOUNDARY_VALUE) / BOUNDARY_VALUE;               
        const double error_rate = (prevError - relative_error) / elapsed;
        prevError = relative_error;
        if(relative_error <= MAX_RELATIVE_ERROR) converged = true;
#endif

        //CUDA_CHECK(cudaDeviceSynchronize()); //Not needed since it's
                                               //handled by the unmap calls 
        CUDA_CHECK(cudaUnbindTexture(texIn));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaBufferEven));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaBufferOdd));
        cudaDeviceSynchronize();       
//RENDER
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);
    
        const int width = glutGet(GLUT_WINDOW_WIDTH);
        const int height = glutGet(GLUT_WINDOW_HEIGHT);
        glViewport(0, 0, width, height);
        //setup OpenGL matrices: no more matrix stack in OpenGL >= 3 core
        //profile, need to compute modelview and projection matrix manually
        const float ratio = width / float(height);
        const glm::mat4 orthoProj = glm::ortho(-ratio, ratio,
                                               -1.0f,  1.0f,
                                                1.0f,  -1.0f);
        const glm::mat4 modelView = glm::mat4(1.0f);
        const glm::mat4 MVP       = orthoProj * modelView;
        glUniformMatrix4fv(mvpID, 1, GL_FALSE, glm::value_ptr(MVP));

        //standard OpenGL core profile rendering
        glActiveTexture(GL_TEXTURE0);        
        glBindTexture(GL_TEXTURE_2D, tex);

        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, quadvbo);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);
       
        glBindBuffer(GL_ARRAY_BUFFER, texbo);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6); 
    
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindTexture(GL_TEXTURE_2D, 0);

        glutSwapBuffers();
        glutPostRedisplay();
        ++step; //next step 
#ifdef FANCY
        std::cout << "\rstep: " << step;
        std::cout.flush();
#else     
        //exit if any timing/error value is NAN or inf
        if(relative_error != relative_error || error_rate != error_rate) {
            std::cout << "\nNaN" << std::endl;
            exit(EXIT_SUCCESS); //EXIT_FAILURE is for execution errors not
                                //for errors related to data
        }
        if(isinf(relative_error) || isinf(error_rate)) {
            std::cout << "\ninf" << std::endl;
            exit(EXIT_SUCCESS); //EXIT_FAILURE is for execution errors not
                                //for errors related to data
        }
        std::cout << "\rstep: " << step 
                  << "   error: " << (100 * relative_error)
                  << " %   speed: " << (100 * error_rate) << " %/s   ";
        std::cout.flush();
#endif 
    }


    if(converged) 
        std::cout << "\nConverged in " 
                  << step << " steps"
                  << "  time: " << totalTime << " s"
                  << std::endl;
                     
//CLEANUP
    glDeleteBuffers(1, &quadvbo);
    glDeleteBuffers(1, &texbo);
    glDeleteTextures(1, &texEven);
    glDeleteTextures(1, &texOdd);

    return 0;
}
