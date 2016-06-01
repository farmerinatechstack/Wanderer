/*
 * Copyright 2014 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.vr.sdk.samples.treasurehunt;

import com.google.vr.sdk.audio.GvrAudioEngine;
import com.google.vr.sdk.base.Eye;
import com.google.vr.sdk.base.GvrActivity;
import com.google.vr.sdk.base.GvrView;
import com.google.vr.sdk.base.HeadTransform;
import com.google.vr.sdk.base.Viewport;

import android.content.Context;
import android.hardware.*;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.opengl.GLES20;
import android.opengl.Matrix;
import android.os.Bundle;
import android.os.Vibrator;
import android.util.Log;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;

/**
 * A Google VR sample application.
 * </p><p>
 * The TreasureHunt scene consists of a planar ground grid and a floating
 * "treasure" cube. When the user looks at the cube, the cube will turn gold.
 * While gold, the user can activate the Carboard trigger, which will in turn
 * randomly reposition the cube.
 */
public class TreasureHuntActivity extends GvrActivity implements GvrView.StereoRenderer, SensorEventListener {

  protected float[] modelCube;
  protected float[] modelPosition;

  private static final String TAG = "TreasureHuntActivity";

  private static final float Z_NEAR = 0.1f;
  private static final float Z_FAR = 100.0f;

  private static final float CAMERA_Z = 0.01f;
  private static final float TIME_DELTA = 0.3f;

  private static final float YAW_LIMIT = 0.12f;
  private static final float PITCH_LIMIT = 0.12f;

  private static final int COORDS_PER_VERTEX = 3;

  // We keep the light always position just above the user.
  private static final float[] LIGHT_POS_IN_WORLD_SPACE = new float[] {0.0f, 2.0f, 0.0f, 1.0f};

  // Convenience vector for extracting the position from a matrix via multiplication.
  private static final float[] POS_MATRIX_MULTIPLY_VEC = {0, 0, 0, 1.0f};

  private static final float MIN_MODEL_DISTANCE = 3.0f;
  private static final float MAX_MODEL_DISTANCE = 7.0f;

  private static final String SOUND_FILE = "cube_sound.wav";

  private final float[] lightPosInEyeSpace = new float[4];

  private FloatBuffer floorVertices;
  private FloatBuffer floorColors;
  private FloatBuffer floorNormals;

  private FloatBuffer cubeVertices;
  private FloatBuffer cubeColors;
  private FloatBuffer cubeFoundColors;
  private FloatBuffer cubeNormals;

  private int cubeProgram;
  private int floorProgram;

  private int cubePositionParam;
  private int cubeNormalParam;
  private int cubeColorParam;
  private int cubeModelParam;
  private int cubeModelViewParam;
  private int cubeModelViewProjectionParam;
  private int cubeLightPosParam;

  private int floorPositionParam;
  private int floorNormalParam;
  private int floorColorParam;
  private int floorModelParam;
  private int floorModelViewParam;
  private int floorModelViewProjectionParam;
  private int floorLightPosParam;

  private float[] camera;
  private float[] view;
  private float[] headView;
  private float[] modelViewProjection;
  private float[] modelView;
  private float[] modelFloor;

  private float[] tempPosition;
  private float[] headRotation;

  private float objectDistance = MAX_MODEL_DISTANCE / 2.0f;
  private float floorDepth = 20f;

  private Vibrator vibrator;

  private GvrAudioEngine gvrAudioEngine;
  private volatile int soundId = GvrAudioEngine.INVALID_ID;

  // Android Tracking Data & Sensors
  private static final float NS2S = 1.0f / 1000000000.0f;
  float[] last_values = null;
  float[] velocity = null;
  float[] position = null;
  float[] acceleration = null;
  long last_timestamp = 0;

  private SensorManager sensorManager;
  private Sensor accSensor;
  private Sensor gravSensor;
  private Sensor magSensor;
  private Sensor orientSensor;
  private WifiManager wifiSensor;

  float[] linAccMeasurements = new float[3];
  float[] gravMeasurements = new float[3];
  float[] magMeasurements = new float[3];
  float[] orientMeasurements = new float[3];
  float[] headRotArray = new float[3];

  private float incrementer = 0.5f;

  /**
   * Converts a raw text file, saved as a resource, into an OpenGL ES shader.
   *
   * @param type The type of shader we will be creating.
   * @param resId The resource ID of the raw text file about to be turned into a shader.
   * @return The shader object handler.
   */
  private int loadGLShader(int type, int resId) {
    String code = readRawTextFile(resId);
    int shader = GLES20.glCreateShader(type);
    GLES20.glShaderSource(shader, code);
    GLES20.glCompileShader(shader);

    // Get the compilation status.
    final int[] compileStatus = new int[1];
    GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compileStatus, 0);

    // If the compilation failed, delete the shader.
    if (compileStatus[0] == 0) {
      Log.e(TAG, "Error compiling shader: " + GLES20.glGetShaderInfoLog(shader));
      GLES20.glDeleteShader(shader);
      shader = 0;
    }

    if (shader == 0) {
      throw new RuntimeException("Error creating shader.");
    }

    return shader;
  }

  /**
   * Checks if we've had an error inside of OpenGL ES, and if so what that error is.
   *
   * @param label Label to report in case of error.
   */
  private static void checkGLError(String label) {
    int error;
    while ((error = GLES20.glGetError()) != GLES20.GL_NO_ERROR) {
      Log.e(TAG, label + ": glError " + error);
      throw new RuntimeException(label + ": glError " + error);
    }
  }

  /**
   * Sets the view to our GvrView and initializes the transformation matrices we will use
   * to render our scene.
   */
  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    initializeGvrView();

    modelCube = new float[16];
    camera = new float[16];
    view = new float[16];
    modelViewProjection = new float[16];
    modelView = new float[16];
    modelFloor = new float[16];
    tempPosition = new float[4];
    // Model first appears directly in front of user.
    modelPosition = new float[] {0.0f, 0.0f, -MAX_MODEL_DISTANCE / 2.0f};
    headRotation = new float[4];
    headView = new float[16];
    vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

    // Initalize Sensors
    sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
    accSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
    gravSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
    magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
    orientSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ORIENTATION);

    sensorManager.registerListener(this, accSensor, SensorManager.SENSOR_DELAY_NORMAL);
    sensorManager.registerListener(this, gravSensor, SensorManager.SENSOR_DELAY_NORMAL);
    sensorManager.registerListener(this, magSensor, SensorManager.SENSOR_DELAY_NORMAL);
    sensorManager.registerListener(this, orientSensor, SensorManager.SENSOR_DELAY_NORMAL);

    wifiSensor = (WifiManager) getSystemService(Context.WIFI_SERVICE);

    // Initialize 3D audio engine.
    gvrAudioEngine = new GvrAudioEngine(this, GvrAudioEngine.RenderingMode.BINAURAL_HIGH_QUALITY);
  }

  private void outputSensorDataToScreen(SensorEvent event) {
    if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
     /* System.arraycopy(event.values, 0, gravMeasurements, 0, 3);

      TextView xLabel= (TextView)findViewById(R.id.gravX);
      TextView yLabel= (TextView)findViewById(R.id.gravY);
      TextView zLabel= (TextView)findViewById(R.id.gravZ);

      xLabel.setText("Gravity X: " + gravMeasurements[0]);
      yLabel.setText("Gravity Y: " + gravMeasurements[1]);
      zLabel.setText("Gravity Z: " + gravMeasurements[2]);
*/
    } else if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
      System.arraycopy(event.values, 0, linAccMeasurements, 0, 3);

      TextView xLabel= (TextView)findViewById(R.id.linAccX);
      TextView yLabel= (TextView)findViewById(R.id.linAccY);
      TextView zLabel= (TextView)findViewById(R.id.linAccZ);

      xLabel.setText("LinAcc X: " + linAccMeasurements[0]);
      yLabel.setText("LinAcc Y: " + linAccMeasurements[1]);
      zLabel.setText("LinAcc Z: " + linAccMeasurements[2]);


      TextView xPosLabel= (TextView)findViewById(R.id.posX);
      TextView yPosLabel= (TextView)findViewById(R.id.posY);
      TextView zPosLabel= (TextView)findViewById(R.id.posZ);

      if (position != null) {
        xPosLabel.setText("posX: 0");
        yPosLabel.setText("posY: " + position[0]);
        zPosLabel.setText("posZ: " + position[2]);
      }

    } else if (event.sensor.getType() == Sensor.TYPE_ORIENTATION) {
      /*System.arraycopy(event.values, 0, orientMeasurements, 0, 3);


     // System.arraycopy(headRotArray, 0, orientMeasurements, 0, 3);
      TextView xLabel= (TextView)findViewById(R.id.orientX);
      TextView yLabel= (TextView)findViewById(R.id.orientY);
      TextView zLabel= (TextView)findViewById(R.id.orientZ);

      xLabel.setText("Orientation X: " + orientMeasurements[0]);
      yLabel.setText("Orientation Y: " + orientMeasurements[1]);
      zLabel.setText("Orientation Z: " + orientMeasurements[2]); */


    } else if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
      System.arraycopy(event.values, 0, magMeasurements, 0, 3);
    }
  }

  @Override
  public void onSensorChanged(SensorEvent sensorEvent) {

    // Output Sensor Data to Screen
    outputSensorDataToScreen(sensorEvent);

    /* Rotate Sensors to Earth Space - Needs Testing
    if (gravMeasurements != null && magMeasurements != null && linAccMeasurements != null) {
      float[] rotationM = new float[16];
      if (SensorManager.getRotationMatrix(rotationM, null, mGrav, mGeoMag) != true) {
        Log.d("Error", "Failed rotation computation");
        return;
      }

      float[] invertRotationM = new float[16];
      if (Matrix.invertM(invertRotationM, 0, rotationM, 0) != true) {
        Log.d("Error", "Failed invert matrix computation");
        return;
      }

      float[] worldAcc = new float[4];
      float[] linAccVals4D = {linAccMeasurements[0], linAccMeasurements[1], linAccMeasurements[2], 0.0f};
      Matrix.multiplyMV(worldAcc, 0, invertRotationM, 0, linAccVals4D, 0);

      Log.d("Acc XYZ", Float.toString(worldAcc[0]) + ", " + Float.toString(worldAcc[1]) + ", " + Float.toString(worldAcc[2]));
    }
    */
    if (sensorEvent.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
      // Use Linear Acceleration for Velocity and Position
      if (last_values != null) {
        float dt = (sensorEvent.timestamp - last_timestamp) * NS2S;
      /* aashna
      for(int i = 0; i < 3; ++i){
        velocity[i] += (sensorEvent.values[i] + last_values[i]) / 2.0 * dt;
        position[i] += velocity[i] * dt;
      }
      */
/*
        if (Math.abs(sensorEvent.values[1]) > 0.5f) {

          velocity[0] -= (sensorEvent.values[1] + last_values[1]) / 2.0 * dt;
          position[0] += (velocity[0] * dt) * 2f;
        }

        if (Math.abs(sensorEvent.values[0]) > 0.5f) {
          acceleration[1] += sensorEvent.values[0];
          velocity[1] += (acceleration[1] * dt);
          position[1] += (velocity[1] * dt + (acceleration[1] / 2.0f * dt * dt));
        }
*/
        if (Math.abs(sensorEvent.values[2]) > 0.5f) {
          acceleration[2] += sensorEvent.values[2];
          velocity[2] += (acceleration[2] * dt);
          position[2] += (velocity[2] * dt + (acceleration[2] / 2.0f * dt * dt));
        }
      } else {
        last_values = new float[3];
        velocity = new float[3];
        position = new float[3];
        acceleration = new float[3];
        velocity[0] = velocity[1] = velocity[2] = 0f;
        position[0] = position[1] = position[2] = 0f;
        acceleration[0] = acceleration[1] = acceleration[2] = 0f;
      }

      System.arraycopy(sensorEvent.values, 0, last_values, 0, 3);
      last_timestamp = sensorEvent.timestamp;
    }
  }

  /**
   * Prepares OpenGL ES before we draw a frame.
   *
   * @param headTransform The head transformation in the new frame.
   */
  @Override
  public void onNewFrame(HeadTransform headTransform) {
    setCubeRotation();

    // Build the camera matrix and apply it to the ModelView.

    // Used for production
    Matrix.setLookAtM(camera, 0,
            position[0], 0f, CAMERA_Z,
            position[0], 0f, 0f,
            0.0f, 1.0f, 0.0f); // position[1]
            // eye, center, up


    /*
    // Used to check the virtual world coordinate system.
    incrementer += 0.02f;
    Matrix.setLookAtM(camera, 0,
            incrementer, 0.0f, CAMERA_Z,
            incrementer, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f);
    // eye, center, up
***/

    headTransform.getHeadView(headView, 0);
    // Update the 3d audio engine with the most recent head rotation.
    headTransform.getQuaternion(headRotation, 0);
    //headTransform.getEulerAngles(headRotArray, 0); //aashna

    gvrAudioEngine.setHeadRotation(
            headRotation[0], headRotation[1], headRotation[2], headRotation[3]);
    // Regular update call to GVR audio engine.
    gvrAudioEngine.update();

    checkGLError("onReadyToDraw");
  }

  @Override
  public void onAccuracyChanged(Sensor sensor, int accuracy) {

  }

  public void initializeGvrView() {
    setContentView(R.layout.common_ui);

    GvrView gvrView = (GvrView) findViewById(R.id.gvr_view);
    gvrView.setEGLConfigChooser(8, 8, 8, 8, 16, 8);

    gvrView.setRenderer(this);
    gvrView.setTransitionViewEnabled(true);
    gvrView.setOnCardboardBackButtonListener(
        new Runnable() {
          @Override
          public void run() {
            onBackPressed();
          }
        });
    setGvrView(gvrView);
  }

  @Override
  public void onPause() {
    gvrAudioEngine.pause();
    sensorManager.unregisterListener(this);
    super.onPause();
  }

  @Override
  public void onResume() {
    super.onResume();
    gvrAudioEngine.resume();
  }

  @Override
  public void onRendererShutdown() {
    Log.i(TAG, "onRendererShutdown");
  }

  @Override
  public void onSurfaceChanged(int width, int height) {
    Log.i(TAG, "onSurfaceChanged");
  }

  /**
   * Creates the buffers we use to store information about the 3D world.
   *
   * <p>OpenGL doesn't use Java arrays, but rather needs data in a format it can understand.
   * Hence we use ByteBuffers.
   *
   * @param config The EGL configuration used when creating the surface.
   */
  @Override
  public void onSurfaceCreated(EGLConfig config) {
    Log.i(TAG, "onSurfaceCreated");
    GLES20.glClearColor(0.1f, 0.1f, 0.1f, 0.5f); // Dark background so text shows up well.

    ByteBuffer bbVertices = ByteBuffer.allocateDirect(WorldLayoutData.CUBE_COORDS.length * 4);
    bbVertices.order(ByteOrder.nativeOrder());
    cubeVertices = bbVertices.asFloatBuffer();
    cubeVertices.put(WorldLayoutData.CUBE_COORDS);
    cubeVertices.position(0);

    ByteBuffer bbColors = ByteBuffer.allocateDirect(WorldLayoutData.CUBE_COLORS.length * 4);
    bbColors.order(ByteOrder.nativeOrder());
    cubeColors = bbColors.asFloatBuffer();
    cubeColors.put(WorldLayoutData.CUBE_COLORS);
    cubeColors.position(0);

    ByteBuffer bbFoundColors =
        ByteBuffer.allocateDirect(WorldLayoutData.CUBE_FOUND_COLORS.length * 4);
    bbFoundColors.order(ByteOrder.nativeOrder());
    cubeFoundColors = bbFoundColors.asFloatBuffer();
    cubeFoundColors.put(WorldLayoutData.CUBE_FOUND_COLORS);
    cubeFoundColors.position(0);

    ByteBuffer bbNormals = ByteBuffer.allocateDirect(WorldLayoutData.CUBE_NORMALS.length * 4);
    bbNormals.order(ByteOrder.nativeOrder());
    cubeNormals = bbNormals.asFloatBuffer();
    cubeNormals.put(WorldLayoutData.CUBE_NORMALS);
    cubeNormals.position(0);

    // make a floor
    ByteBuffer bbFloorVertices = ByteBuffer.allocateDirect(WorldLayoutData.FLOOR_COORDS.length * 4);
    bbFloorVertices.order(ByteOrder.nativeOrder());
    floorVertices = bbFloorVertices.asFloatBuffer();
    floorVertices.put(WorldLayoutData.FLOOR_COORDS);
    floorVertices.position(0);

    ByteBuffer bbFloorNormals = ByteBuffer.allocateDirect(WorldLayoutData.FLOOR_NORMALS.length * 4);
    bbFloorNormals.order(ByteOrder.nativeOrder());
    floorNormals = bbFloorNormals.asFloatBuffer();
    floorNormals.put(WorldLayoutData.FLOOR_NORMALS);
    floorNormals.position(0);

    ByteBuffer bbFloorColors = ByteBuffer.allocateDirect(WorldLayoutData.FLOOR_COLORS.length * 4);
    bbFloorColors.order(ByteOrder.nativeOrder());
    floorColors = bbFloorColors.asFloatBuffer();
    floorColors.put(WorldLayoutData.FLOOR_COLORS);
    floorColors.position(0);

    int vertexShader = loadGLShader(GLES20.GL_VERTEX_SHADER, R.raw.light_vertex);
    int gridShader = loadGLShader(GLES20.GL_FRAGMENT_SHADER, R.raw.grid_fragment);
    int passthroughShader = loadGLShader(GLES20.GL_FRAGMENT_SHADER, R.raw.passthrough_fragment);

    cubeProgram = GLES20.glCreateProgram();
    GLES20.glAttachShader(cubeProgram, vertexShader);
    GLES20.glAttachShader(cubeProgram, passthroughShader);
    GLES20.glLinkProgram(cubeProgram);
    GLES20.glUseProgram(cubeProgram);

    checkGLError("Cube program");

    cubePositionParam = GLES20.glGetAttribLocation(cubeProgram, "a_Position");
    cubeNormalParam = GLES20.glGetAttribLocation(cubeProgram, "a_Normal");
    cubeColorParam = GLES20.glGetAttribLocation(cubeProgram, "a_Color");

    cubeModelParam = GLES20.glGetUniformLocation(cubeProgram, "u_Model");
    cubeModelViewParam = GLES20.glGetUniformLocation(cubeProgram, "u_MVMatrix");
    cubeModelViewProjectionParam = GLES20.glGetUniformLocation(cubeProgram, "u_MVP");
    cubeLightPosParam = GLES20.glGetUniformLocation(cubeProgram, "u_LightPos");

    checkGLError("Cube program params");

    floorProgram = GLES20.glCreateProgram();
    GLES20.glAttachShader(floorProgram, vertexShader);
    GLES20.glAttachShader(floorProgram, gridShader);
    GLES20.glLinkProgram(floorProgram);
    GLES20.glUseProgram(floorProgram);

    checkGLError("Floor program");

    floorModelParam = GLES20.glGetUniformLocation(floorProgram, "u_Model");
    floorModelViewParam = GLES20.glGetUniformLocation(floorProgram, "u_MVMatrix");
    floorModelViewProjectionParam = GLES20.glGetUniformLocation(floorProgram, "u_MVP");
    floorLightPosParam = GLES20.glGetUniformLocation(floorProgram, "u_LightPos");

    floorPositionParam = GLES20.glGetAttribLocation(floorProgram, "a_Position");
    floorNormalParam = GLES20.glGetAttribLocation(floorProgram, "a_Normal");
    floorColorParam = GLES20.glGetAttribLocation(floorProgram, "a_Color");

    checkGLError("Floor program params");

    Matrix.setIdentityM(modelFloor, 0);
    Matrix.translateM(modelFloor, 0, 0, -floorDepth, 0); // Floor appears below user.

    // Avoid any delays during start-up due to decoding of sound files.
    new Thread(
            new Runnable() {
              @Override
              public void run() {
                // Start spatial audio playback of SOUND_FILE at the model postion. The returned
                //soundId handle is stored and allows for repositioning the sound object whenever
                // the cube position changes.
                gvrAudioEngine.preloadSoundFile(SOUND_FILE);
                soundId = gvrAudioEngine.createSoundObject(SOUND_FILE);
                gvrAudioEngine.setSoundObjectPosition(
                    soundId, modelPosition[0], modelPosition[1], modelPosition[2]);
                gvrAudioEngine.playSound(soundId, true /* looped playback */);
              }
            })
        .start();

    updateModelPosition();

    checkGLError("onSurfaceCreated");
  }

  /**
   * Updates the cube model position.
   */
  protected void updateModelPosition() {
    Matrix.setIdentityM(modelCube, 0);
    Matrix.translateM(modelCube, 0, modelPosition[0], modelPosition[1], modelPosition[2]);

    // Update the sound location to match it with the new cube position.
    if (soundId != GvrAudioEngine.INVALID_ID) {
      gvrAudioEngine.setSoundObjectPosition(
          soundId, modelPosition[0], modelPosition[1], modelPosition[2]);
    }
    checkGLError("updateCubePosition");
  }

  /**
   * Converts a raw text file into a string.
   *
   * @param resId The resource ID of the raw text file about to be turned into a shader.
   * @return The context of the text file, or null in case of error.
   */
  private String readRawTextFile(int resId) {
    InputStream inputStream = getResources().openRawResource(resId);
    try {
      BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
      StringBuilder sb = new StringBuilder();
      String line;
      while ((line = reader.readLine()) != null) {
        sb.append(line).append("\n");
      }
      reader.close();
      return sb.toString();
    } catch (IOException e) {
      e.printStackTrace();
    }
    return null;
  }

  protected void setCubeRotation() {
    Matrix.rotateM(modelCube, 0, TIME_DELTA, 0.5f, 0.5f, 1.0f);
  }

  /**
   * Draws a frame for an eye.
   *
   * @param eye The eye to render. Includes all required transformations.
   */
  @Override
  public void onDrawEye(Eye eye) {
    GLES20.glEnable(GLES20.GL_DEPTH_TEST);
    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

    checkGLError("colorParam");

    // Apply the eye transformation to the camera.
    Matrix.multiplyMM(view, 0, eye.getEyeView(), 0, camera, 0);

    // Set the position of the light
    Matrix.multiplyMV(lightPosInEyeSpace, 0, view, 0, LIGHT_POS_IN_WORLD_SPACE, 0);

    // Build the ModelView and ModelViewProjection matrices
    // for calculating cube position and light.
    float[] perspective = eye.getPerspective(Z_NEAR, Z_FAR);
    Matrix.multiplyMM(modelView, 0, view, 0, modelCube, 0);
    Matrix.multiplyMM(modelViewProjection, 0, perspective, 0, modelView, 0);
    drawCube();

    // Set modelView for the floor, so we draw floor in the correct location
    Matrix.multiplyMM(modelView, 0, view, 0, modelFloor, 0);
    Matrix.multiplyMM(modelViewProjection, 0, perspective, 0, modelView, 0);
    drawFloor();
  }

  @Override
  public void onFinishFrame(Viewport viewport) {}

  /**
   * Draw the cube.
   *
   * <p>We've set all of our transformation matrices. Now we simply pass them into the shader.
   */
  public void drawCube() {
    GLES20.glUseProgram(cubeProgram);

    GLES20.glUniform3fv(cubeLightPosParam, 1, lightPosInEyeSpace, 0);

    // Set the Model in the shader, used to calculate lighting
    GLES20.glUniformMatrix4fv(cubeModelParam, 1, false, modelCube, 0);

    // Set the ModelView in the shader, used to calculate lighting
    GLES20.glUniformMatrix4fv(cubeModelViewParam, 1, false, modelView, 0);

    // Set the position of the cube
    GLES20.glVertexAttribPointer(
        cubePositionParam, COORDS_PER_VERTEX, GLES20.GL_FLOAT, false, 0, cubeVertices);

    // Set the ModelViewProjection matrix in the shader.
    GLES20.glUniformMatrix4fv(cubeModelViewProjectionParam, 1, false, modelViewProjection, 0);

    // Set the normal positions of the cube, again for shading
    GLES20.glVertexAttribPointer(cubeNormalParam, 3, GLES20.GL_FLOAT, false, 0, cubeNormals);
    GLES20.glVertexAttribPointer(cubeColorParam, 4, GLES20.GL_FLOAT, false, 0,
        isLookingAtObject() ? cubeFoundColors : cubeColors);

    // Enable vertex arrays
    GLES20.glEnableVertexAttribArray(cubePositionParam);
    GLES20.glEnableVertexAttribArray(cubeNormalParam);
    GLES20.glEnableVertexAttribArray(cubeColorParam);

    GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 36);
    checkGLError("Drawing cube");
  }

  /**
   * Draw the floor.
   *
   * <p>This feeds in data for the floor into the shader. Note that this doesn't feed in data about
   * position of the light, so if we rewrite our code to draw the floor first, the lighting might
   * look strange.
   */
  public void drawFloor() {
    GLES20.glUseProgram(floorProgram);

    // Set ModelView, MVP, position, normals, and color.
    GLES20.glUniform3fv(floorLightPosParam, 1, lightPosInEyeSpace, 0);
    GLES20.glUniformMatrix4fv(floorModelParam, 1, false, modelFloor, 0);
    GLES20.glUniformMatrix4fv(floorModelViewParam, 1, false, modelView, 0);
    GLES20.glUniformMatrix4fv(floorModelViewProjectionParam, 1, false, modelViewProjection, 0);
    GLES20.glVertexAttribPointer(
        floorPositionParam, COORDS_PER_VERTEX, GLES20.GL_FLOAT, false, 0, floorVertices);
    GLES20.glVertexAttribPointer(floorNormalParam, 3, GLES20.GL_FLOAT, false, 0, floorNormals);
    GLES20.glVertexAttribPointer(floorColorParam, 4, GLES20.GL_FLOAT, false, 0, floorColors);

    GLES20.glEnableVertexAttribArray(floorPositionParam);
    GLES20.glEnableVertexAttribArray(floorNormalParam);
    GLES20.glEnableVertexAttribArray(floorColorParam);

    GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 24);

    checkGLError("drawing floor");
  }

  /**
   * Called when the Cardboard trigger is pulled.
   */
  @Override
  public void onCardboardTrigger() {
    Log.i(TAG, "onCardboardTrigger");

    if (isLookingAtObject()) {
      hideObject();
    }

    // Always give user feedback.
    vibrator.vibrate(50);
  }

  /**
   * Find a new random position for the object.
   *
   * <p>We'll rotate it around the Y-axis so it's out of sight, and then up or down by a little bit.
   */
  protected void hideObject() {
    float[] rotationMatrix = new float[16];
    float[] posVec = new float[4];

    // First rotate in XZ plane, between 90 and 270 deg away, and scale so that we vary
    // the object's distance from the user.
    float angleXZ = (float) Math.random() * 180 + 90;
    Matrix.setRotateM(rotationMatrix, 0, angleXZ, 0f, 1f, 0f);
    float oldObjectDistance = objectDistance;
    objectDistance =
        (float) Math.random() * (MAX_MODEL_DISTANCE - MIN_MODEL_DISTANCE) + MIN_MODEL_DISTANCE;
    float objectScalingFactor = objectDistance / oldObjectDistance;
    Matrix.scaleM(rotationMatrix, 0, objectScalingFactor, objectScalingFactor, objectScalingFactor);
    Matrix.multiplyMV(posVec, 0, rotationMatrix, 0, modelCube, 12);

    float angleY = (float) Math.random() * 80 - 40; // Angle in Y plane, between -40 and 40.
    angleY = (float) Math.toRadians(angleY);
    float newY = (float) Math.tan(angleY) * objectDistance;

    modelPosition[0] = posVec[0];
    modelPosition[1] = newY;
    modelPosition[2] = posVec[2];

    updateModelPosition();
  }

  /**
   * Check if user is looking at object by calculating where the object is in eye-space.
   *
   * @return true if the user is looking at the object.
   */
  private boolean isLookingAtObject() {
    // Convert object space to camera space. Use the headView from onNewFrame.
    Matrix.multiplyMM(modelView, 0, headView, 0, modelCube, 0);
    Matrix.multiplyMV(tempPosition, 0, modelView, 0, POS_MATRIX_MULTIPLY_VEC, 0);

    float pitch = (float) Math.atan2(tempPosition[1], -tempPosition[2]);
    float yaw = (float) Math.atan2(tempPosition[0], -tempPosition[2]);

    return Math.abs(pitch) < PITCH_LIMIT && Math.abs(yaw) < YAW_LIMIT;
  }
}
