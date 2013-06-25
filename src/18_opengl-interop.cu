//OpenGL-CUDA interop
//Author: Ugo Varetto


///////////////// IN PROGRESS ////////////////////

//Requires GLFWT and GLM, to deal with the missing support for matrix stack
//in OpenGL >= 3.3

//nvcc ../src/18_cuda-opengl.cu \
// ../src/gl-cl.cpp -I/usr/local/glfw/include \
// -DGL_GLEXT_PROTOTYPES -L/usr/local/glfw/lib -lglfw \
// -I/usr/local/cuda/include -lOpenCL \
// -I/usr/local/glm/include

#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath> //isinf

#include <cuda_gl_interop.h>

#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


#define gle std::cout << "[GL] - " \
                      << __LINE__ << ' ' << glGetError() << std::endl;

//globals required by CUDA: since CUDA does not allow direct texture write, it
//is not possible to simply have a kernel that receives an input and output
//texture as easily done in OpenCL; cuda requires access to textures(input)
//and surfaces(output) bound toarray and declared as global variables only

//Mapping goes like this:
//OpenGL texture -->
//  CUDA Graphics resource -->
//    CUDA Array -->
//      CUDA Texture [IN]
//      CUDA Surface [OUT] !!!


texture<float,2>  texIn; //read 
surface<float,2>  surfaceOut; //write

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
    for(int y = 0; y != height; ++y) {
        for(int x = 0; x != width; ++x) {
            if(y < yOffset
               || x < xOffset
               || y >= height - yOffset
               || x >= width - xOffset) g[y * width + x] = value;
            else g[y * width + x] = float(0);
        }
    }
    return g;
}

//------------------------------------------------------------------------------
void error_callback(int error, const char* description) {
    std::cerr << description << std::endl;
}

//------------------------------------------------------------------------------
void key_callback(GLFWwindow* window, int key,
                  int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}


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
   // surf2DWrite TODO
}


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
    "  color = vec3(smoothstep(minvalue, maxvalue, c));\n"
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

//------------------------------------------------------------------------------
bool IS_EVEN(int v) { return v % 2 == 0; }

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
//USER INPUT
    if(argc > 1 && argc < 5) {
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
    try {
        const int STENCIL_SIZE = 3;
        const int SIZE = argc > 1 ? atoi(argv[2]) : 34;
        const int THREADS_PER_BLOCK = argc > 1 ? atoi(argv[3]) : 4;
        const int BLOCKS = (SIZE - 2 * (STENCIL_SIZE / 2)) / THREADS_PER_BLOCK;
        const float DIFFUSION_SPEED = argc > 1 ? atof(argv[4]) : 0.22;
        const float BOUNDARY_VALUE = argc > 5 ? atof(argv[5]) : 1.0f;
//GRAPHICS SETUP        
        glfwSetErrorCallback(error_callback);

        if(!glfwInit()) {
            std::cerr << "ERROR - glfwInit" << std::endl;
            exit(EXIT_FAILURE);
        }

        GLFWwindow* window = glfwCreateWindow(640, 480,
                                              "OpenCL/GL interop", NULL, NULL);
        if(!window) {
            std::cerr << "ERROR - glfwCreateWindow" << std::endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        
        glfwSetKeyCallback(window, key_callback);
   
        glfwMakeContextCurrent(window);     
      

//GEOMETRY AND OPENCL-OPENGL MAPPING
 
        //geometry: textured quad; the texture color value is computed by
        //OpenCL
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
        glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float),
                     &quad[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        GLuint texbo;  
        glGenBuffers(1, &texbo);
        glBindBuffer(GL_ARRAY_BUFFER, texbo);
        glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float),
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
        cudaGraphicsResource* cudaBufferEven = 0;
        cudaGraphicsGLRegisterImage(&cudaBufferEven, texEven, GL_TEXTURE_2D,
                                    cudaGraphicsMapFlagsNone);
        cudaGraphicsResource* cudaBufferOdd = 0;
        cudaGraphicsGLRegisterImage(&cudaBufferOdd, texOdd, GL_TEXTURE_2D,
                                    cudaGraphicsMapFlagsNone); 
      

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
        double start = glfwGetTime();
        double totalTime = 0;
       
        float prevError = 0;
        while (!glfwWindowShouldClose(window) && !converged) {     

//COMPUTE AND CHECK CONVERGENCE           
            glFinish(); //<-- ensure Open*G*L is done

            cudaArray* arrayIn = 0;
            cudaArray* arrayOut = 0;
            cudaGraphicsMapResources(1, &cudaBufferEven);
            cudaGraphicsMapResources(1, &cudaBufferOdd);
               
            if(IS_EVEN(step)) {
                cudaGraphicsSubResourceGetMappedArray(&arrayIn, cudaBufferEven,
                                                      0, 0);
                cudaGraphicsSubResourceGetMappedArray(&arrayOut, cudaBufferOdd,
                                                      0, 0);
                tex = texOdd;
            } else {//even
                cudaGraphicsSubResourceGetMappedArray(&arrayOut, cudaBufferEven,
                                                      0, 0);
                cudaGraphicsSubResourceGetMappedArray(&arrayIn, cudaBufferOdd,
                                                      0, 0);
                tex = texEven;
            }
            cudaBindTextureToArray(texIn, arrayIn);
            cudaBindSurfaceToArray(surfaceOut, arrayOut);
            
            apply_stencil<<<BLOCKS, THREADS_PER_BLOCK>>>(DIFFUSION_SPEED);
            // //CHECK FOR CONVERGENCE: extract element at grid center
            // //and exit if |element value - boundary value| <= EPS    
            // float centerOut = -BOUNDARY_VALUE;
            // int activeBuffer = IS_EVEN(step) ? 1 : 0;
            // cl::size_t<3> origin;
            // origin[0] = SIZE / 2;
            // origin[1] = SIZE / 2;
            // origin[2] = 0;
            // cl::size_t<3> region;
            // region[0] = 1;
            // region[1] = 1;
            // region[2] = 1;
            // queue.enqueueReadImage(clbuffers[activeBuffer],
            //                        CL_TRUE,
            //                        origin,
            //                        region,
            //                        0, //row pitch; zero for delegating
            //                           //computation to OpenCL
            //                        0, //slice pitch: for 3D only
            //                        &centerOut);
            // const double elapsed = glfwGetTime() - start;
            // totalTime += elapsed;
            // start = elapsed;
            // const float MAX_RELATIVE_ERROR = 0.01;//%
            // const float relative_error =
            //     fabs(centerOut - BOUNDARY_VALUE) / BOUNDARY_VALUE;
            // if(step == 0) prevError = relative_error;
                
            // const double error_rate = -(relative_error - prevError) / elapsed;
            // if(relative_error <= MAX_RELATIVE_ERROR) converged = true;
            cudaDeviceSynchronize(); //<-- ensure CUDA is done  
            cudaUnbindTexture(texIn);
            cudaGraphicsUnmapResources(1, &cudaBufferEven);
            cudaGraphicsUnmapResources(1, &cudaBufferOdd);
                   
//RENDER
            // Clear the screen
            glClear(GL_COLOR_BUFFER_BIT);
        
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
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
            glfwSwapBuffers(window);
            glfwPollEvents();

            ++step; //next step 
            //printout if all values are not NAN
            // if(relative_error != relative_error || error_rate != error_rate) {
            //     std::cout << "\nNaN" << std::endl;
            //     exit(EXIT_SUCCESS); //EXIT_FAILURE is for execution errors not
            //                         //for errors related to data
            // }
            // //if any value is inf do exit
            // if(isinf(relative_error) || isinf(error_rate)) {
            //     std::cout << "\ninf" << std::endl;
            //     exit(EXIT_SUCCESS); //EXIT_FAILURE is for execution errors not
            //                         //for errors related to data
            // }
            // std::cout << "\rstep: " << step 
            //           << "   error: " << (100 * relative_error)
            //           << " %   speed: " << (100 * error_rate) << " %/s   ";
            // std::cout.flush();
        }

        if(converged) 
            std::cout << "\nConverged in " 
                      << step << " steps"
                      << "  time: " << totalTime / 1E3 << " s"
                      << std::endl;
//CLEANUP
        glDeleteBuffers(1, &quadvbo);
        glDeleteBuffers(1, &texbo);
        glDeleteTextures(1, &texEven);
        glDeleteTextures(1, &texOdd);
        glfwDestroyWindow(window);

        //TODO: DELETE CUDA RESOURCES

        glfwTerminate();
        exit(EXIT_SUCCESS);
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
