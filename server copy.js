const express = require('express');
const https = require('https');
const bcrypt = require('bcrypt');
const { exec } = require('child_process');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const app = express();
const bodyParser = require('body-parser');
const Docker = require('dockerode');
const docker = new Docker();


// The default splatName to use if a user doesn't specify it
DEFAULT_SPLAT_NAME = "ericPoggies";
var splatName = DEFAULT_SPLAT_NAME
var splatNameList = []

// Code to set up and run docker container
const preprocessImageName = 'preprocess';
const trainingImageName = 'train';
const renderImageName = 'nerfstudio';

const condaTrainingEnvName = "gaussian_splatting";
const numIterations = 250;


// The path to the images that are initially uploaded
const imageFolderToClear = 'public/uploads/model_input';
const imageFolderPath = 'public/uploads/model_input/';
const TMP_UPLOADS_PATH = 'public/tmp_uploads';
const ZIP_FOLDER_PATH = 'public/raw_device_uploads/';
const transformsFolderPath = 'public/app_camera_transforms_folder/';
const DEFAULT_APP_CAMERA_POSE_NAME = 'app_camera_poses.json';
const DEFAULT_BOUNDING_BOX_NAME = 'boundingbox.json';
const DEFAULT_CAMERAS_NAME = "cameras.json";
const DEFAULT_CHANGED_CAMERAS_NAME = "original_cameras.json";


const boundingboxFolderPath = 'public/boundingbox_folder/';


// Mounting goes <path on host:path in container>
// Image output paths
const containerInputFolderPath = "/gaussian-splatting/public/"; //startingContainerPath + endingContainerPath;
const renderStartingContainerPath = '/nerfstudio_gaussviewer/nerfstudio/';

const containerTrainedOutputPath = '/gaussian-splatting/output/';


const statusTypes = {
    WAITING_FOR_DATA: 'waiting_for_data',
    PREPROCESSING_STARTED: 'preprocessing_started',
    PREPROCESSING_ENDED: 'preprocessing_ended',
    TRAINING_STARTED: 'training_started',
    TRAINING_ENDED: 'training_ended',
    RENDERING_STARTED: 'rendering_started',
    RENDERING_ENDED: 'rendering_ended',
}

let status = statusTypes.WAITING_FOR_DATA;

/**
 * The variables preprocessingFilePathInContainer & trainFilePathInContainer below are used to find the 
 * files for preprocessing our input images with colmap and training our gaussian splatting model respectively.
 * These files are in the container. The variable pythonScriptsFolderPath may need to be changed in the future
 * if when we get into the container we are in a different folder
 */
const pythonScriptsFolderPath = '';
const convertFileName = 'convert.py';
const trainFileName = 'train.py';
const preprocessingFilePathInContainer = pythonScriptsFolderPath + convertFileName;
const trainFilePathInContainer = pythonScriptsFolderPath + trainFileName;


// Preprocessing output paths
const containerModelProcessedOutputPath = "";
var postProcessedImagesFolderPath = "public/model_output/post_processing" + "/" + splatName;

// Training output paths
const containerModelTrainOutputFolderPath = ""; // Have to set this to wherever the model gets outputted in the container 
var trainedSplatTempFolderPath = "public/nerfstudio_gaussviewer/nerfstudio/model_output/post_train" + "/" + splatName;
var trainedSplatFolderPath = "public/model_output/post_train" + "/" + splatName;
var containerRenderInputFolderPath = '/nerfstudio_gaussviewer/nerfstudio/model_output/post_train/' + splatName + '/output/' + splatName; // '/nerfstudio_gaussviewer/nerfstudio/model_output/post_train/output' + "/" + splatName;

var renderedOutputFolderPath = "public/model_output/post_render/" + splatName + "/";

var preprocessInputFolderPath = "public/uploads/model_input/" + splatName + "/";

// <host_path> : <container_path>
var preprocessMountPath = path.join(__dirname, preprocessInputFolderPath + ':' + containerInputFolderPath);
var trainingMountPath = path.join(__dirname, postProcessedImagesFolderPath + ":" + containerInputFolderPath);
var renderMountPath = path.join(__dirname, 'public/nerfstudio_gaussviewer:/nerfstudio_gaussviewer');

// Create the command to convert
var commandToTrainModel = 'python ' + trainFilePathInContainer + ' -s ' + containerInputFolderPath + ' --model_path ./output/' + splatName + ' --test_iterations ' + numIterations + ' --save_iterations ' + numIterations + ' --stop_iteration ' + numIterations;

// Commands to run in the container 
const preprocessCommand = 'python3 ' + preprocessingFilePathInContainer + ' -s ' + containerInputFolderPath;

// This command preprocess the images found inside the input folder in the docker container relative to the directory we arrive at. The relative path is: 
// public/uploads/model_input. We assume we land in the docker container at /server/gaussian-splatting
// This command is used in conjunction with mounting. The files in the server folder "public/uploads/model_input" are placed into the docker 
// container at ./server/gaussian-splatting/public/uploads/model_input
// The folder ./server/gaussian-splatting/public/uploads/model_input will the container all of the processed folders and files. After running this command
// in the runPreprocessCommand function below, we copy the folder ./server/gaussian-splatting/public/uploads/model_input into the server 


// This command trains the model on the processed images found inside the input folder in the docker container relative to the directory we arrive at.
// Where this command is used, we first mount the processed folders and files into the docker containre at the location ./server/gaussian-splatting/public/uploads/model_input.
// We assume we arrive in the docker container at /server/gaussian-splatting. We then run the command to train the model based on the folders and files in the 
// relative folder /public/uploads/model_input
var trainCommand = 'conda init bash && echo "about to exec bash" &&  exec bash -c "source activate ' + condaTrainingEnvName + ' && ' + commandToTrainModel + '"';

const privateKey = fs.readFileSync('public/open-ssl/server.ky', 'utf8');
const certificate = fs.readFileSync('public/open-ssl/server.cert', 'utf8');
const credentials = { key: privateKey, cert: certificate };

const httpsServer = https.createServer(credentials, app);
const IMAGE_PORT = 15001; // 8000;
httpsServer.listen(IMAGE_PORT, '0.0.0.0', () => {
    console.log(`Server is running on https://localhost:${IMAGE_PORT}`)
});

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json())

function changePathsForSplatName(splatName) {
    preprocessInputFolderPath = "public/uploads/model_input/" + splatName + "/";
    postProcessedImagesFolderPath = "public/model_output/post_processing" + "/" + splatName;
    preprocessMountPath = path.join(__dirname, preprocessInputFolderPath + ':' + containerInputFolderPath);

    trainingMountPath = path.join(__dirname, postProcessedImagesFolderPath + ":" + containerInputFolderPath);
    trainedSplatTempFolderPath = "public/nerfstudio_gaussviewer/nerfstudio/model_output/post_train" + "/" + splatName;
    trainedSplatFolderPath = "public/model_output/post_train" + "/" + splatName;
    commandToTrainModel = 'python ' + trainFilePathInContainer + ' -s ' + containerInputFolderPath + ' --model_path ./output/' + splatName + ' --test_iterations ' + numIterations + ' --save_iterations ' + numIterations + ' --stop_iteration ' + numIterations;
    trainCommand = 'conda init bash && echo "about to exec bash" &&  exec bash -c "source activate ' + condaTrainingEnvName + ' && ' + commandToTrainModel + '"';

    containerRenderInputFolderPath = '/nerfstudio_gaussviewer/nerfstudio/model_output/post_train/' + splatName + '/output/' + splatName; // '/nerfstudio_gaussviewer/nerfstudio/model_output/post_train/output' + "/" + splatName;
    renderedOutputFolderPath = "public/model_output/post_render/" + splatName + "/";
}

// I'm pretty sure this function is good to go
function runPreprocessCommand() {
    imageName = preprocessImageName;
    mountPath = preprocessMountPath;
    containerCommand = preprocessCommand;
    containerFolderToSavePath = containerInputFolderPath + "."; // We need the . so that it doesn't save the public directory, but everything in the directory
    localSavePath = postProcessedImagesFolderPath; // We add the splatName to the path to save 

    console.log("image name: " + imageName);
    console.log("mountPath: " + mountPath);
    const containerOptions = {
        Image: imageName,
        name: 'preprocess',
        Cmd: ['/bin/bash'], // Specify the command to run within the container
        Tty: true,
        AttachStdin: true,
        AttachStdout: true,
        AttachStderr: true,
        OpenStdin: true,
        HostConfig: {
            AutoRemove: true,
            Runtime: 'nvidia',
            Binds: [`${mountPath}`] // We have to put the mountPath here of HostConfig
        },
    };

    // Create the container
    docker.createContainer(containerOptions, function (err, container) {
        if (err) {
            console.error(err);
            return;
        } else {
            // Trouble shoots why we get the 409 error
            container.inspect((err, data) => {
                if (err) {
                    console.error('Error inspecting container:', err);
                } else {
                    const isRunning = data.State.Running;

                    if (isRunning) {
                        console.log('Container is already running.');
                    } else {
                        // Start the container here
                        container.start((startErr) => {
                            if (startErr) {
                                console.error('Error starting container:', startErr);
                            } else {
                                console.log('Container started successfully.');
                            }
                        });
                    }
                }
            });

            // Get the containerId
            const containerId = container.id;

            // Start the container
            container.start(function (err, data) {
                if (err) {
                    console.error(err);
                    return;
                }

                // Set the command to execute
                console.log("containerCommand: " + containerCommand);
                // containerCommand = "ls public/uploads/model_input/input";
                const execOptions =
                {
                    Cmd: ['/bin/bash', '-c', containerCommand],
                    AttachStdout: true,
                    AttachStderr: true,
                };

                // Execute that command
                container.exec(execOptions, function (err, exec1) {
                    if (err) {
                        console.error(err);
                        return;
                    }

                    exec1.start(function (err, stream) {
                        if (err) {
                            console.error(err);
                            return;
                        }

                        container.modem.demuxStream(stream, process.stdout, process.stderr);

                        stream.on('end', function () {
                            // console.log('Preprocess command executed, about to save');
                            console.log("SAVING");
                            refreshFolderSync(localSavePath);
                            const saveOutputCommand = 'docker cp ' + container.id + ':' + containerFolderToSavePath + ' ' + localSavePath;


                            // Save the output of the preprocess command
                            exec(saveOutputCommand, (error, stdout, stderr) => {
                                if (error) {
                                    console.error(`Error executing the command: ${error.message}`);
                                    return;
                                }

                                if (stderr) {
                                    console.error(`Command execution produced an error: ${stderr}`);
                                    return;
                                }

                                console.log(`Command output preprocess: ${stdout}`);
                                // Run training container
                                runTrainCommand();
                            });
                            // Stop the container
                            container.stop(function (err, data) {
                                if (err) {
                                    console.error('Error stopping preprocessing container:', err);
                                } else {
                                    console.log('Preprocessing container stopped successfully:', data);
                                }
                            });
                            return containerId;
                        });
                    });
                });
            });
        }
    });
}

function runTrainCommand() {

    imageName = trainingImageName;
    mountPath = trainingMountPath;
    containerCommand = trainCommand;
    containerFolderToSavePath = containerTrainedOutputPath;
    localSavePath = trainedSplatTempFolderPath;
    serverSavePath = trainedSplatFolderPath;

    console.log("image name: " + imageName);
    console.log("mountPath: " + mountPath);
    const containerOptions = {
        Image: imageName,
        name: 'train',
        Cmd: ['/bin/bash'], // Specify the command to run within the container
        Tty: true,
        AttachStdin: true,
        AttachStdout: true,
        AttachStderr: true,
        OpenStdin: true,
        HostConfig: {
            AutoRemove: true,
            Runtime: 'nvidia',
            Binds: [`${mountPath}`] // We have to put the mountPath here of HostConfig
        },
    };

    // Create the container
    docker.createContainer(containerOptions, function (err, container) {
        if (err) {
            console.error(err);
            return;
        } else {
            // Get the containerId
            const containerId = container.id;

            // Start the container
            container.start(function (err, data) {
                if (err) {
                    console.error(err);
                    return;
                }

                // Set the command to execute
                console.log("containerCommand: " + containerCommand);
                // containerCommand = "ls public/uploads/model_input/input";
                const execOptions =
                {
                    Cmd: ['/bin/bash', '-c', containerCommand],
                    AttachStdout: true,
                    AttachStderr: true,
                };

                // Execute that command
                container.exec(execOptions, function (err, exec1) {
                    if (err) {
                        console.error(err);
                        return;
                    }

                    exec1.start(function (err, stream) {
                        if (err) {
                            console.error(err);
                            return;
                        }

                        container.modem.demuxStream(stream, process.stdout, process.stderr);

                        stream.on('end', function () {
                            // console.log('Preprocess command executed, about to save');
                            console.log("SAVING");
                            refreshFolderSync(serverSavePath);
                            refreshFolderSync(localSavePath);

                            const saveOutputCommandOnServer = 'docker cp ' + container.id + ':' + containerFolderToSavePath + ' ' + serverSavePath;
                            const saveOutputCommand = 'docker cp ' + container.id + ':' + containerFolderToSavePath + ' ' + localSavePath;

                            exec(saveOutputCommandOnServer, (error, stdout, stderr) => {
                                if (error) {
                                    console.error(`Error executing the command: ${error.message}`);
                                    return;
                                }

                                if (stderr) {
                                    console.error(`Command execution produced an error: ${stderr}`);
                                    return;
                                }
                                console.log(`Command output train: ${stdout}`);
                                // Save the output of the preprocess command into nerfstudio_gaussiviewer/nerfstudio
                                exec(saveOutputCommand, (error, stdout, stderr) => {
                                    if (error) {
                                        console.error(`Error executing the command: ${error.message}`);
                                        return;
                                    }

                                    if (stderr) {
                                        console.error(`Command execution produced an error: ${stderr}`);
                                        return;
                                    }
                                    // // First we want to rename the file 'cameras.json' to 'input_camera_path.json'
                                    // renameCameraFile();
                                    // Second we want to create the camera_path file and name it cameras.json
                                    createCameraPathFile()
                                        .then(() => {
                                            runRenderCustomPathCommand();
                                        })
                                        .catch(error=> {
                                            console.error('Error:', error);
                                        });
                                    // Then we want to create a bounding box in the model's frame 
                                    // Then we want to make a new .ply file and delete the one in the folder 
                                    // Run the render command after the trained output has saved
                                    // The command for the next step always has to be inside of the exec so it is gauranteed to run after 
                                    
                                    console.log(`Command output train executing rest: ${stdout}`);
                                });
                            });

                            // Stop the container
                            container.stop(function (err, data) {
                                if (err) {
                                    console.error('Error stopping train container:', err);
                                } else {
                                    console.log('Train container stopped successfully:', data);
                                }

                            });
                            return containerId;
                        });
                    });
                });
            });
        }
    });
}

function runRenderCommand() {
    const imageName = renderImageName;
    const mountPath = renderMountPath;
    const renderCommand = 'python nerfstudio/scripts/gaussian_splatting/render.py interpolate --model-path ' + containerRenderInputFolderPath + ' --pose-source train --output-path ' + renderedOutputFolderPath + 'output_video.mp4';
    const containerCommand = "cd /nerfstudio_gaussviewer/nerfstudio ; pip install ./submodules/diff-gaussian-rasterization ./submodules/simple-knn ; " + renderCommand;
    const containerFolderToSavePath = renderStartingContainerPath + renderedOutputFolderPath;
    const localSavePath = renderedOutputFolderPath;

    console.log("image name: " + imageName);
    console.log("mountPath: " + mountPath);
    const containerOptions = {
        Image: imageName,
        name: 'render',
        Cmd: ['/bin/bash'], // Specify the command to run within the container
        Tty: true,
        AttachStdin: true,
        AttachStdout: true,
        AttachStderr: true,
        OpenStdin: true,
        HostConfig: {
            AutoRemove: true,
            Runtime: 'nvidia',
            ShmSize: 12 * 1024 * 1024 * 1024,
            PortBindings: {
                '7007/tcp': [{ HostPort: '7007' }],
            },
            Binds: [`${mountPath}`] // We have to put the mountPath here of HostConfig
        },
    };

    // Create the container
    docker.createContainer(containerOptions, function (err, container) {
        if (err) {
            console.error(err);
            return;
        } else {
            // Get the containerId
            const containerId = container.id;

            // Start the container
            container.start(function (err, data) {
                if (err) {
                    console.error(err);
                    return;
                }

                // Set the command to execute
                console.log("containerCommand: " + containerCommand);
                // containerCommand = "ls public/uploads/model_input/input";
                const execOptions =
                {
                    Cmd: ['/bin/bash', '-c', containerCommand],
                    AttachStdout: true,
                    AttachStderr: true,
                };

                // Execute that command
                container.exec(execOptions, function (err, exec1) {
                    if (err) {
                        console.error(err);
                        return;
                    }

                    exec1.start(function (err, stream) {
                        if (err) {
                            console.error(err);
                            return;
                        }

                        container.modem.demuxStream(stream, process.stdout, process.stderr);

                        stream.on('end', function () {
                            // docker cp <container_id_or_name>:<container_path> <local_path>
                            console.log("SAVING");
                            refreshFolderSync(localSavePath);
                            const saveOutputCommand = 'docker cp ' + container.id + ':' + containerFolderToSavePath + ' ' + localSavePath;


                            // Save the output of the preprocess command
                            exec(saveOutputCommand, (error, stdout, stderr) => {
                                if (error) {
                                    console.error(`Error executing the command: ${error.message}`);
                                    return;
                                }

                                if (stderr) {
                                    console.error(`Command execution produced an error: ${stderr}`);
                                    return;
                                }

                                console.log(`Command output regular render: ${stdout}`);
                            });
                            // Stop the container
                            container.stop(function (err, data) {
                                if (err) {
                                    console.error('Error stopping render container:', err);
                                } else {
                                    console.log('Render container stopped successfully:', data);
                                }
                            });
                            return containerId;
                        });
                    });
                });
                return containerId;
            });
        }
    });
}

function runRenderCustomPathCommand() {
    const imageName = renderImageName;
    const mountPath = renderMountPath;
    // const renderCommand = 'python nerfstudio/scripts/gaussian_splatting/render.py camera-path --model-path ' + containerRenderInputFolderPath + ' --camera-path-filename /nerfstudio_gaussviewer/nerfstudio/camera_path.json --output-path ' + renderedOutputFolderPath + 'output_video.mp4';
    const renderCommand = 'python nerfstudio/scripts/gaussian_splatting/render.py interpolate --model-path ' + containerRenderInputFolderPath + ' --pose-source train --output-path ' + renderedOutputFolderPath + 'output_video.mp4';
    
    const containerCommand = "cd /nerfstudio_gaussviewer/nerfstudio ; pip install ./submodules/diff-gaussian-rasterization ./submodules/simple-knn ; " + renderCommand;
    const containerFolderToSavePath = renderStartingContainerPath + renderedOutputFolderPath + "."; // We must add the "." so that we only save what is in the directory
    const localSavePath = renderedOutputFolderPath;

    console.log("image name: " + imageName);
    console.log("mountPath: " + mountPath);
    const containerOptions = {
        Image: imageName,
        name: 'render',
        Cmd: ['/bin/bash'], // Specify the command to run within the container
        Tty: true,
        AttachStdin: true,
        AttachStdout: true,
        AttachStderr: true,
        OpenStdin: true,
        HostConfig: {
            AutoRemove: true,
            Runtime: 'nvidia',
            ShmSize: 12 * 1024 * 1024 * 1024,
            PortBindings: {
                '7007/tcp': [{ HostPort: '7007' }],
            },
            Binds: [`${mountPath}`] // We have to put the mountPath here of HostConfig
        },
    };

    // Create the container
    docker.createContainer(containerOptions, function (err, container) {
        if (err) {
            console.error(err);
            return;
        } else {
            // Get the containerId
            const containerId = container.id;

            // Start the container
            container.start(function (err, data) {
                if (err) {
                    console.error(err);
                    return;
                }

                // Set the command to execute
                console.log("containerCommand: " + containerCommand);
                // containerCommand = "ls public/uploads/model_input/input";
                const execOptions =
                {
                    Cmd: ['/bin/bash', '-c', containerCommand],
                    AttachStdout: true,
                    AttachStderr: true,
                };

                // Execute that command
                container.exec(execOptions, function (err, exec1) {
                    if (err) {
                        console.error(err);
                        return;
                    }

                    exec1.start(function (err, stream) {
                        if (err) {
                            console.error(err);
                            return;
                        }

                        container.modem.demuxStream(stream, process.stdout, process.stderr);

                        stream.on('end', function () {
                            // docker cp <container_id_or_name>:<container_path> <local_path>
                            console.log("SAVING");
                            refreshFolderSync(localSavePath);
                            const saveOutputCommand = 'docker cp ' + container.id + ':' + containerFolderToSavePath + ' ' + localSavePath;

                            // Save the output of the preprocess command
                            exec(saveOutputCommand, (error, stdout, stderr) => {
                                if (error) {
                                    console.error(`Error executing the command: ${error.message}`);
                                    return;
                                }

                                if (stderr) {
                                    console.error(`Command execution produced an error: ${stderr}`);
                                    return;
                                }

                                console.log(`Command output custom render: ${stdout}`);
                            });
                            // Stop the container
                            container.stop(function (err, data) {
                                if (err) {
                                    console.error('Error stopping render container:', err);
                                } else {
                                    console.log('Render container stopped successfully:', data);
                                }
                            });
                            return containerId;
                        });
                    });
                });
                return containerId;
            });
        }
    });
}

function getSubdirectoryName(mainDirectory) {
    // Read the contents of folder1
    const files = fs.readdirSync(mainDirectory);

    // Filter out only directories
    const subdirectories = files.filter(file => fs.statSync(`${mainDirectory}/${file}`).isDirectory());

    if (subdirectories.length === 1) {
        // There is exactly one subdirectory
        var subdirectoryName = subdirectories[0];
        console.log('The name of folder2 is:', subdirectoryName);
        return subdirectoryName;
    } else {
        console.error('There are either zero or multiple subdirectories in folder1.');
        return "-1";
    }
}

function getAllSubdirectoryNames(mainDirectory) {
    // Read the contents of folder1
    const files = fs.readdirSync(mainDirectory);

    // Filter out only directories
    const subdirectories = files.filter(file => fs.statSync(`${mainDirectory}/${file}`).isDirectory());

    return subdirectories
}

function deleteFilesInDirectory(directory) {
    // Synchronously delete all files and subdirectories within folderA
    fs.readdirSync(directory).forEach(item => {
        const itemPath = path.join(directory, item);
        if (fs.statSync(itemPath).isDirectory()) {
            // Delete directory recursively
            fs.rmdirSync(itemPath, { recursive: true });
        } else {
            // Delete file
            fs.unlinkSync(itemPath);
        }
    });
}

function createDirectory(directory) {
    // Create a new folder within folderA
    fs.mkdirSync(directory);
}

function refreshFolderSync(folderPath) {
    console.log("refreshFolderSync called");
    // Check if the folder exists
    if (fs.existsSync(folderPath)) {
        deleteFilesInDirectory(folderPath);
        console.log(`Folder created at ${folderPath}`);
    } else {
        // Create the folder if it doesn't exist
        const directories = folderPath.split(path.sep);
        let currentPath = '';
        // Iterate through each directory and create it if it doesn't exist
        for (const directory of directories) {
            currentPath = path.join(currentPath, directory);
            if (!fs.existsSync(currentPath)) {
                fs.mkdirSync(currentPath);
            }
        }
        console.log(`Folder created at ${folderPath}`);
    }
}


function createCameraPathFile() {
    return new Promise((resolve, reject) => {
        const sourceFilePath = path.join(__dirname, trainedSplatFolderPath + "/output/" + splatName + "/" + DEFAULT_CAMERAS_NAME);
        const destinationFilePath = path.join(__dirname, trainedSplatFolderPath + "/output/" + splatName + "/" + DEFAULT_CHANGED_CAMERAS_NAME);
        console.log("sourceFilePath: " + sourceFilePath);
        console.log("destinationFilePath: " + destinationFilePath);
        console.log("About to call fs.renameSync");
        fs.renameSync(sourceFilePath, destinationFilePath, (err) => {
            if (err) {
                console.error('Error renaming file:', err);
                reject();
                return res.status(500).send('Error renaming file');
            }
            const boundingBoxFilePath = path.join(__dirname, 'public/boundingbox_folder/' + splatName + "/" + DEFAULT_BOUNDING_BOX_NAME);
            const appCameraPosesFilePath = path.join(__dirname,  transformsFolderPath + splatName + "/" + DEFAULT_APP_CAMERA_POSE_NAME);
            const modelCameraPosesFilePath = path.join(__dirname, trainedSplatFolderPath + "/output/" + splatName + "/" + DEFAULT_CHANGED_CAMERAS_NAME);
            const outputPath = path.join(__dirname, trainedSplatFolderPath + "/output/" + splatName + "/" + DEFAULT_CAMERAS_NAME);
            const cameraPythonScriptPath = path.join(__dirname, 'public/scripts/get_camera_poses.py');
            const cameraPathCommand = 'pip install open3d ; python3 ' + cameraPythonScriptPath + ' ' + boundingBoxFilePath + ' ' + appCameraPosesFilePath + ' ' + modelCameraPosesFilePath + ' ' + outputPath + ' 100';

            console.log("Dir name: " + __dirname);
            console.log("boundingBoxFilePath name: " + boundingBoxFilePath);
            console.log("outputPath name: " + outputPath);
            console.log("Camera path command: " + cameraPathCommand);

            exec(cameraPathCommand, (error, stdout, stderr) => {
                if (error) {
                    console.error(`Error executing the command: ${error.message}`);
                    reject();
                    return;
                }

                if (stderr) {
                    console.error(`Command execution produced an error: ${stderr}`);
                    reject();
                    return;
                }
                console.log(`Command output creatingCameraPath: ${stdout}`);

                console.log('createCamerePath function has ran');

                resolve();
            });
            // runRenderCustomPathCommand();
            console.log('Files uploaded and renamed successfully');
        });
    });
}

const PORT = 15002;
//process.env.PORT || 3000;

app.use(express.static("public"))
app.set('view engine', 'ejs');


app.get('/', (req, res) => {
    // refreshFolderSync("public/model_output/") // I use this to clear model_output folder when testing
    splatNameList = getAllSubdirectoryNames(imageFolderPath);
    res.render("upload", { splatNameList: splatNameList })
});


 // ------ Webviewer ------
 // JSON file with data on rendered splats
const objectData = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'objects.json'), 'utf-8'));

// Read HTML template (antimatter webviewer)
const template = fs.readFileSync(path.join(__dirname, 'views', 'index.ejs'), 'utf-8');
// Route for each object page
objectData.forEach(obj => {
    console.log(`Registering route: /object/${obj.id}`);
    app.get(`object/${obj.id}`, (req, res) => {
        const htmlContent = template.replace('{{title}}', obj.title)
                                    .replace('{{splatPathParam}}', `?splatPath=${encodeURIComponent(obj.splatPath)}`);
        res.send(htmlContent);
    });
});


app.get('/download', (req, res) => {
    const splatPath = req.query.splatPath; // Get splatPath from the query string
    const filePath = path.join(__dirname, splatPath);

    res.setHeader('Content-Disposition', `attachment; filename=${path.basename(splatPath)}`);
    //res.setHeader('Content-Type', 'text/plain');

    res.sendFile(filePath);
})

const galleryTemplate = fs.readFileSync(path.join(__dirname, 'views', 'gallery.ejs'), 'utf-8');
app.get('/webviewer', (req, res) => {
    // Generate gallery items for each object
    const galleryItems = objectData.map(obj =>
        `<div class="item">
            <h2 class="item-title">${obj.title}</h2>
            <a href="/object/${obj.id}">Closer Look</a>
            <img src="${obj.imagePath}"></img>
        </div>`
    ).join('');
    
    // Insert gallery items into gallery template
    const galleryContent = galleryTemplate.replace('{{galleryItems}}', galleryItems);

    res.send(galleryContent);
});



// app.get('/index', (req, res) => {
//     res.render('index.ejs')
// });



// Route to handle image upload 
app.post('/upload', (req, res) => {
    const allowedExtensions = ['.jpg', '.png'];
    console.log("Splat name: " + splatName);
    const splatUploadResetFolder = imageFolderPath + splatName;
    const splatInputImageFolderPath = imageFolderPath + splatName + "/input";
    refreshFolderSync(splatUploadResetFolder);

    // Create a storage engine for multer to save uploaded images 
    const storage = multer.diskStorage({
        destination: path.join(__dirname, splatInputImageFolderPath),
        filename: function (req, file, cb) {
            cb(null, Date.now() + '-' + file.originalname);
        },
    });


    // Initialize multer with the storage engine 
    const upload = multer({ storage });

    upload.any()(req, res, (err) => {
        if (err) {
            return res.status(400).send('File upload failed! D:');
        }

        const validFiles = req.files.filter(file => {
            const fileExtension = path.extname(file.originalname.toLowerCase());
            return allowedExtensions.includes(fileExtension)
        })

        // You can perform further processing here 
        if (validFiles.length === 0) {
            return res.status(400).send('No images were uploaded.');
        }

        // Loop through uplaoded files
        validFiles.forEach((file) => {
            console.log(`Uploaded file: ${file.originalname}`);
        });

        res.send('Images uploaded successfully.');
    });
});

// Path to the temporary uploads directory
const tmpUploadsDir = path.join(__dirname, TMP_UPLOADS_PATH);

// Ensure the temporary uploads directory exists
if (!fs.existsSync(tmpUploadsDir)){
    fs.mkdirSync(tmpUploadsDir, { recursive: true });
    console.log('Temporary uploads directory created at:', tmpUploadsDir);
} else {
    console.log('Temporary uploads directory already exists:', tmpUploadsDir);
}

// Temporary storage setup
const tmpStorage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, tmpUploadsDir); // Temporary directory
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname); // Use the original file name
  }
});

const tmp_upload = multer({ storage: tmpStorage });

// Route to handle uploads
app.post('/upload_device_data', tmp_upload.array('files'), (req, res) => {
  const splatName = req.body.splatName; // The folder name specified in the request body

  // Ensure the target directory exists
  const targetDir = path.join(__dirname, ZIP_FOLDER_PATH, splatName);
  if (!fs.existsSync(targetDir)){
    fs.mkdirSync(targetDir, { recursive: true });
  }

  // Move files from the temporary directory to the target directory
  req.files.forEach(file => {
    const tempPath = file.path;
    const targetPath = path.join(targetDir, file.originalname);

    fs.rename(tempPath, targetPath, function(err) {
      if (err) throw err;
      console.log(`Successfully moved ${file.originalname} to ${targetFolder}`);
    });
  });

  res.send('Files uploaded successfully');
});



// app.post('/upload_device_data', (req, res) => {
//     // const allowedExtensions = ['.jpg', '.png'];
//     // console.log("Splat name: " + splatName);
//     // const splatUploadResetFolder = imageFolderPath + splatName;
//     // const splatInputImageFolderPath = imageFolderPath + splatName + "/input";
//     // refreshFolderSync(splatUploadResetFolder);
//     splatName = req.splatName;
//     const splatZipPath = ZIP_FOLDER_PATH+splatName

//     // Create a storage engine for multer to save uploaded images 
//     const storage = multer.diskStorage({
//         destination: path.join(__dirname, splatInputImageFolderPath),
//         filename: function (req, file, cb) {
//             cb(null, Date.now() + '-' + file.originalname);
//         },
//     });


//     // Initialize multer with the storage engine 
//     const upload = multer({ storage });

//     upload.any()(req, res, (err) => {
//         if (err) {
//             return res.status(400).send('File upload failed! D:');
//         }

//         const validFiles = req.files.filter(file => {
//             const fileExtension = path.extname(file.originalname.toLowerCase());
//             return allowedExtensions.includes(fileExtension)
//         })

//         // You can perform further processing here 
//         if (validFiles.length === 0) {
//             return res.status(400).send('No images were uploaded.');
//         }

//         // Loop through uplaoded files
//         validFiles.forEach((file) => {
//             console.log(`Uploaded file: ${file.originalname}`);
//         });

//         res.send('Images uploaded successfully.');
//     });
// });

app.post('/upload_splat_name', (req, res) => {
    splatName = req.body.splatName;
    changePathsForSplatName(splatName);
    splatNameList = getAllSubdirectoryNames(imageFolderPath);
    res.send("Splat name recorded as: " + splatName);
});

app.post('/upload_boundingbox', (req, res) => {
    const boundingBoxUploadResetFolder = boundingboxFolderPath + splatName;
    refreshFolderSync(boundingBoxUploadResetFolder);
    const allowedExtensions = ['.json'];
    // Clear the uploads folder before processing new uploads
    const uploadsDir = path.join(__dirname, boundingBoxUploadResetFolder);// imageFolderToClear);
    fs.readdir(uploadsDir, (err, files) => {
        if (err) {
            return console.error('Error reading uploads folder bounding box:', err);
        }
        for (const file of files) {
            fs.unlink(path.join(uploadsDir, file), (err) => {
                if (err) {
                    return console.error(`Error deleting file ${file.filename}:`, err);
                }
            });
        }
    });

    // Create a storage engine for multer to save uploaded images 
    const bboxStorage = multer.diskStorage({
        destination: path.join(__dirname, boundingBoxUploadResetFolder),
        filename: function (req, file, cb) {
            cb(null, DEFAULT_BOUNDING_BOX_NAME);
        },
    });

    const bboxUpload = multer({ storage: bboxStorage });

    bboxUpload.any()(req, res, (err) => {
        if (err) {
            console.log(err);
            return res.status(400).send('File upload failed! D:');
        }

        const validFiles = req.files.filter(file => {
            const fileExtension = path.extname(file.originalname.toLowerCase());
            return allowedExtensions.includes(fileExtension)
        })

        // You can perform further processing here 
        if (validFiles.length === 0) {
            return res.status(400).send('No bounding box was uploaded.');
        }

        // Loop through uplaoded files
        validFiles.forEach((file) => {
            console.log(`Bounding box Uploaded file: ${file.originalname}`);
        });

        // // Now we want to create the camera path script 
        // createCameraPathFile();
        res.send('Bounding box uploaded successfully?');
    });
});

app.post('/upload_transforms', (req, res) => {
    const allowedExtensions = ['.json'];
    console.log("Splat name: " + splatName);
    const transformsUploadResetFolder = transformsFolderPath + splatName;
    // const splatInputImageFolderPath = imageFolderPath + splatName + "/input";
    refreshFolderSync(transformsUploadResetFolder);

    // Create a storage engine for multer to save uploaded images 
    const storage = multer.diskStorage({
        destination: path.join(__dirname, transformsUploadResetFolder),
        filename: function (req, file, cb) {
            cb(null, DEFAULT_APP_CAMERA_POSE_NAME);
        },
    });


    // Initialize multer with the storage engine 
    const upload = multer({ storage });

    upload.any()(req, res, (err) => {
        if (err) {
            return res.status(400).send('File upload failed! D:');
        }

        const validFiles = req.files.filter(file => {
            const fileExtension = path.extname(file.originalname.toLowerCase());
            return allowedExtensions.includes(fileExtension)
        })

        // You can perform further processing here 
        if (validFiles.length === 0) {
            return res.status(400).send('No camera poses were uploaded.');
        }

        // Loop through uplaoded files
        validFiles.forEach((file) => {
            console.log(`Uploaded file: ${file.originalname} as ${DEFAULT_APP_CAMERA_POSE_NAME}`);
        });

        res.send('App camera poses uploaded successfully.');
    });
});

app.post('/full_pipeline', (req, res) => {
    // The preprocess command starts the pieline
    runPreprocessCommand();
    res.send("Pipeline Starting");
});

app.post('/preprocess', (req, res) => {
    status = statusTypes.PREPROCESSING_STARTED;
    runPreprocessCommand();
    status = statusTypes.PREPROCESSING_ENDED;
    res.send("Images Processing Attempted");
});

app.post('/train', (req, res) => {
    status = statusTypes.TRAINING_STARTED;
    runTrainCommand();
    status = statusTypes.TRAINING_ENDED;
    res.send("Model Training Attempted");

});

app.post('/render', (req, res) => {
    status = statusTypes.RENDERING_STARTED;
    runRenderCommand();
    status = statusTypes.RENDERING_ENDED;
    res.send("Model Rendered Attempted");
});

app.post('/reset_status', (req, res) => {
    status = statusTypes.WAITING_FOR_DATA;
    res.send({"status": status});
})

app.get('/status', (req, res) => {
    res.send({"status": status});
})

app.post('/render_custom_path', (req, res) => {
    status = statusTypes.RENDERING_STARTED;
    runRenderCustomPathCommand();
    status = statusTypes.RENDERING_ENDED;
    res.send("Model Custom Render Attempted");
});

app.get('/heartbeat', (req, res) => {
    res.json({ message: "Hello World" });
});

// app.listen(PORT, () => { 
//     console.log(`Server is running on port ${PORT}`); 
// });

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT}`);
});