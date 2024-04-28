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
const admZip = require('adm-zip');
const mongoose = require('mongoose');

// The default splatName to use if a user doesn't specify it
var splatNameList = [];

// This is used if you need to refresh splats immediately when the server starts up
var onStart = 1;

// The directory where we store the rendered videos
const videosDirectory = "public/model_output/post_render";
// The names of the docker images that we had created
const preprocessImageName = 'preprocess';
const trainingImageName = 'train';
const renderImageName = 'nerfstudio';

// The conda environment where we have a lot of libraries imported that we use for background culling and camera path creation 
const condaTrainingEnvName = "gaussian_splatting";
// The number of iterations that the 3D gaussian splatting model runs for 
const numIterations = 4000; 

// Folder path to save depth images
const depthFolderPath = 'public/depth_images';
// Folder path to save the initially uploaded images
const imageFolderPath = 'public/uploads/model_input/';

// Folder path to save the transforms data from the app 
// about the poses of where pictures were taken from the app's frame
const transformsFolderPath = 'public/app_camera_transforms_folder/';
// Just for temp files that aren't used anywhere else, used in device uploads
const TMP_UPLOADS_PATH = 'public/tmp_uploads';
// Name we use for the camera poses we get from the app
const DEFAULT_APP_CAMERA_POSE_NAME = 'app_camera_poses.json';
// Name we use for the bounding box in the app's frame
const DEFAULT_APP_BOUNDING_BOX_NAME = 'boundingbox.json';
// The name we use for the bounding box in the model's frame
const DEFAULT_MODEL_BOUNDING_BOX_NAME = 'model_boundingbox.json';
// Name of the cameras.json file in the output of 3D gaussian splatting. 
// This is produced by colmap and contains colmap's estimates of the poses of
// where the camera positions were taken. We use this to create the transformation
// from the app's coordinate frame to colmap's (the model's) coordinate frame
const DEFAULT_CAMERAS_NAME = "cameras.json";
// Contains the content of the original cameras.json file. We replace the content
// in cameras.json to be the camera path so we can use nerf-studio's interpolate 
// function which is hard coded to use a file named "cameras.json" 
const DEFAULT_CHANGED_CAMERAS_NAME = "original_cameras.json";
// This is where the initial 3D gaussian splat is saved but we replace this 
// with the culled version of the splat
const DEFAULT_POINT_CLOUD_NAME = "point_cloud.ply"
// Contains the original splat points without culling
const DEFAULT_CHANGED_POINT_CLOUD_NAME = "original_point_cloud.ply"
const BOUNDING_BOX_FOLDER = 'public/boundingbox_folder/';
const BOUNDING_BOX_FOLDER_NAME = 'boundingbox_folder';
const DEFAULT_VIDEO_NAME = 'output_video.mp4';

// If we are using mongo. This is an artifact from before where
// had a different way of getting filepaths
const usingMongo = true;

// Used to let us upload to the temporary uploads directory 
const tmpUploadsDir = path.join(__dirname, TMP_UPLOADS_PATH);
// Ensure the temporary uploads directory exists
if (!fs.existsSync(tmpUploadsDir)){
    fs.mkdirSync(tmpUploadsDir, { recursive: true });
    console.log('Temporary uploads directory created at:', tmpUploadsDir);
} else {
    console.log('Temporary uploads directory already exists:', tmpUploadsDir);
}


// Mounting goes <path on host:path in container>
// Paths that we use in each of the docker images
const containerInputFolderPath = "/gaussian-splatting/public/"; 
const renderStartingContainerPath = '/nerfstudio_gaussviewer/nerfstudio/';
const containerTrainedOutputPath = '/gaussian-splatting/output/';

// We send this info back to the app so it knows what step we are on
const statusTypes = {
    WAITING_FOR_DATA: 'waiting_for_data',
    DATA_UPLOAD_STARTED: 'data_upload_started',
    DATA_UPLOAD_ENDED: 'data_upload_ended',
    PREPROCESSING_STARTED: 'preprocessing_started',
    PREPROCESSING_ENDED: 'preprocessing_ended',
    TRAINING_STARTED: 'training_started',
    TRAINING_ENDED: 'training_ended',
    RENDERING_STARTED: 'rendering_started',
    RENDERING_ENDED: 'rendering_ended',

    DATA_UPLOAD_ERROR: 'data_upload_error',
    PREPROCESSING_ERROR: 'preprocessing_error',
    TRAINING_ERROR: 'training_error',
    RENDERING_ERROR: 'rendering_error'
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

// Left over from when we used a .json file to keep track of file paths
const webviewerSplatJson = "data/objects.json";

// Mounting works like
// <host_path> : <container_path>

// Commands to run in the container 
const preprocessCommand = 'python3 ' + preprocessingFilePathInContainer + ' -s ' + containerInputFolderPath;
const renderMountPath = path.join(__dirname, 'public/nerfstudio_gaussviewer:/nerfstudio_gaussviewer');

// Connect to MongoDB
const MONGODB_HOST = "localhost";
const MONGODB_PORT = "27017";
const database_name = "GaussianSplatInfo";
const dbURI = `mongodb://${MONGODB_HOST}:${MONGODB_PORT}/${database_name}`;
console.log("dbURI: " + dbURI);
mongoose.connect(dbURI, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(result => console.log("database connected"));
const db = mongoose.connection;
db.on('error', console.error.bind(console, 'MongoDB connection error:'));

// Define schema
const Schema = mongoose.Schema;

const dataSchema = new Schema({
    name: { type: String, required: true, unique: true },
    preprocessInputFolderPath: { type: String, required: true },
    postProcessedImagesFolderPath: { type: String, required: true },
    preprocessMountPath: { type: String, required: true },
    trainingMountPath: { type: String, required: true },
    trainedSplatTempFolderPath: { type: String, required: true },
    trainedSplatFolderPath: { type: String, required: true },
    commandToTrainModel: { type: String, required: true },
    trainCommand: { type: String, required: true },
    containerRenderInputFolderPath: { type: String, required: true },
    renderedOutputFolderPath: { type: String, required: true },
    webviewerSplatFolderPath: { type: String, required: true },
    webviewerSplatFilePath: { type: String, required: true },
    webviewerImagePath: {type: String},
    webviewerVideoPath: {type: String},
    webviewerLink: {type: String},
    pointCloudOriginalNameFilePath: {type: String},
    pointCloudNewNameFilePath: {type: String},
    modelBoundingBoxFilePath: {type: String},
    modelBoundingBoxRelativeFilePath: {type: String},
    appBoundingBoxFilePath: {type: String},
    appCameraPosesFilePath: {type: String},
    modelCameraPosesFilePath: {type: String}
});

// Define model
const DataModel = mongoose.model('Data', dataSchema);

// We should not be getting duplicate key errors anymore
// Below will implement duplicate key error code 
dataSchema.post('save', function(error, doc, next) {
    if (error.code === 11000) {
        // Handle the duplicate key error
        console.errror("Duplicate key error: ", error);
    } else {
        // Pass the error to the next middleware
        console.log(error);
    }
});

const privateKey = fs.readFileSync('public/open-ssl/server.ky', 'utf8');
const certificate = fs.readFileSync('public/open-ssl/server.cert', 'utf8');
const credentials = { key: privateKey, cert: certificate };

const httpsServer = https.createServer(credentials, app);
const IMAGE_PORT = 15001;
httpsServer.listen(IMAGE_PORT, '0.0.0.0', () => {
    console.log(`Server is running on https://localhost:${IMAGE_PORT}`)
});

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json())

/**
 * Currently, we don't use this because it assumes that all the splats 
 * in the render folder have the same number of iterations which messes
 * up some file paths and thus makes this not work. 
 * @param {*} splatList 
 * @returns 
 */
function createMultipleSplatsInDB(splatList) {
    // SplatList should be a list of all the rendered splats.

    // Creating a new data entry
    return new Promise ((resolve, reject) => {
        getNumSplatsInJson()
        .then((numSplats) => {
            var count = 0;
            for (let splatNumber = numSplats; splatNumber < splatList.length+numSplats; splatNumber++) {
                var splatName = splatList[count];
                
                createNewSplatDataOnDB(splatName)
                .then((_) => {
                    console.log("Splat created for: " + splatName);
                })
                .catch((err) => {
                    console.log("Error calling create new splat from create multiple splats: ", err);
                });
                count++;

            }
            resolve();
        });
    });
}

/**
 * Returns true or false on whether a splat is in the database
 * @param {String} splatName 
 * @returns {Boolean} 
 */
function isSplatInDB(splatName) {
    return new Promise((resolve, reject) => {
        findSplatInDB(splatName) 
        .then((foundSplat) => {
            if (foundSplat == null) {
                resolve(false);
            } else {
                resolve(true);
            }
        })
        .catch((err) => {
            console.error("Error in isSplatInDB in findSplatInDB: ", err);
            reject();
        })
    });
}

/**
 * Creates a new splat in the data base. If the splat is already in the database, 
 * then we refresh it's paths to be up to date with the current server variables.
 * Currently, this just changes the numIterations variable that is present in 
 * many of the file paths
 * @param {String} splatName 
 * @returns {JSON} JSON of the deleted splat
 */
function createNewSplatDataOnDB(splatName) {
    // Example usage:
    // Creating a new data entry
    return new Promise ((resolve, reject) => {
        isSplatInDB(splatName) 
        .then((splatExists) => {
            if (splatExists) {
                refreshSplatInDB(splatName)
                .then((savedData) => {
                    resolve(savedData);
                })
                .catch((err) => {
                    reject();
                });
            } else {
                getNumSplatsInJson()
                .then((numSplats) => {
                    console.log("About to create splat with name: ", splatName);
                    var imageFolderPath = "/uploads/model_input/" + splatName + "/input/";
                    const newData = new DataModel({
                        name: splatName,
                        preprocessInputFolderPath: 'public/uploads/model_input/' + splatName + '/',
                        postProcessedImagesFolderPath: 'public/model_output/post_processing/' + splatName,
                        preprocessMountPath: path.join(__dirname, 'public/uploads/model_input/' + splatName + '/' + ':' + containerInputFolderPath),
                        trainingMountPath: path.join(__dirname, 'public/model_output/post_processing/' + splatName + ':' + containerInputFolderPath),
                        trainedSplatTempFolderPath: 'public/nerfstudio_gaussviewer/nerfstudio/model_output/post_train/' + splatName,
                        trainedSplatFolderPath: 'public/model_output/post_train/' + splatName,
                        commandToTrainModel: 'python ' + trainFilePathInContainer + ' -s ' + containerInputFolderPath + ' --model_path ./output/' + splatName + ' --test_iterations ' + numIterations + ' --save_iterations ' + numIterations + ' --stop_iteration ' + numIterations,
                        trainCommand: 'conda init bash && echo "about to exec bash" && exec bash -c "source activate ' + condaTrainingEnvName + ' && ' + 'python ' + trainFilePathInContainer + ' -s ' + containerInputFolderPath + ' --model_path ./output/' + splatName + ' --test_iterations ' + numIterations + ' --save_iterations ' + numIterations + ' --stop_iteration ' + numIterations + '"',
                        containerRenderInputFolderPath: '/nerfstudio_gaussviewer/nerfstudio/model_output/post_train/' + splatName + '/output/' + splatName,
                        renderedOutputFolderPath: 'public/model_output/post_render/' + splatName + '/',
                        webviewerSplatFolderPath: 'public/splat/' + splatName,
                        webviewerSplatFilePath: 'public/splat/' + splatName + '/output.splat',
                        webviewerImagePath: imageFolderPath + getFirstPNGFileName("public" + imageFolderPath),
                        webviewerVideoPath: path.join("/model_output/post_render/", splatName, DEFAULT_VIDEO_NAME),
                        pointCloudOriginalNameFilePath: path.join(__dirname, 'public/model_output/post_train/' + splatName + "/output/" + splatName + "/point_cloud/iteration_" + numIterations + "/" + DEFAULT_POINT_CLOUD_NAME),
                        pointCloudNewNameFilePath: path.join(__dirname, 'public/model_output/post_train/' + splatName + "/output/" + splatName + "/point_cloud/iteration_" + numIterations + "/" + DEFAULT_CHANGED_POINT_CLOUD_NAME),
                        modelBoundingBoxFilePath: path.join(__dirname, BOUNDING_BOX_FOLDER + splatName + "/" + DEFAULT_MODEL_BOUNDING_BOX_NAME),
                        modelBoundingBoxRelativeFilePath: path.join("/", BOUNDING_BOX_FOLDER_NAME, splatName + "/" + DEFAULT_MODEL_BOUNDING_BOX_NAME),
                        appBoundingBoxFilePath: path.join(__dirname, BOUNDING_BOX_FOLDER + splatName + "/" + DEFAULT_APP_BOUNDING_BOX_NAME),
                        appCameraPosesFilePath: path.join(__dirname,  transformsFolderPath + splatName + "/" + DEFAULT_APP_CAMERA_POSE_NAME),
                        modelCameraPosesFilePath: path.join(__dirname, 'public/model_output/post_train/' + splatName + "/output/" + splatName + "/" + DEFAULT_CHANGED_CAMERAS_NAME)
                    });
                    newData.save()
                    .then(function(splatJson) {
                        console.log(" Creating new splat for database: " + splatJson["name"] + " " + splatJson["_id"]);
                        // console.log(splatJson["name"]);
                        // const filter = { name: splatName };
                        // const update = {webviewerLink: "http://osiris.cs.hmc.edu:15002/object/" + splatJson["_id"]};
                        // DataModel.findOneAndUpdate(filter, update);

                        // We have to add the webviewer link after we have made the object because we use the database hash to find the object in the webviewer
                        const update = {webviewerLink: "http://osiris.cs.hmc.edu:15002/object/" + splatJson["_id"]};
                        updateSplat(splatName, update)
                        .then((splatJson) => {
                            console.log("New splat created with updated link");
                            resolve(splatJson);
                        })
                        .catch((err) => {
                            console.log("Splat created but error updating with new webviewer link: ", err);
                        });
                        
                    })
                    .catch(function(err) {
                        console.error(err);
                        reject();
                    });
                })
                .catch((err) => {
                    console.log("Error in createNewSplatInDB in getNumSplats json: ", err);
                    reject();
                });
            }
        })
        .catch((err)=> {
            console.error("Error in createNewSplatDataOnDB in findSplatInDB: ", err);
            reject();
        })
        
    });
    
}

/**
 * Deletes a splat in the database
 * @param {String} splatName 
 * @returns {JSON} JSON of the deleted Splat
 */
function deleteSplatInDB(splatName) {
    return new Promise ((resolve, reject) => {
        DataModel.deleteOne({name: splatName})
        .then((splatJson) => {
            resolve(splatJson);
        })
        .catch((error) => {
            console.error("Deleting Item Error: ", error);
            reject(null);
        });
    });
}

/**
 * Deletes all the splats in the database. USE WITH CAUTION!
 * @returns {Array} Array of all the splats that were deleted
 */
function deleteAllSplatsInDB() {
    return new Promise ((resolve, reject) => {
        DataModel.deleteMany({})
        .then((allSplats) => {
            resolve(allSplats);
        })
        .catch((error) => {
            console.error("Deleting Item Error: ", error);
            reject(null);
        });
    });
}

/**
 * Refreshes a splat in the data base by using findOneAndUpdate. This assumes
 * the splat is already there, so do this checking before the function
 * @param {String} splatName 
 * @returns {JSON} The updated splat as a JSON 
 */
function refreshSplatInDB(splatName) {    
    return new Promise ((resolve, reject) => {
        findSplatInDB(splatName)
        .then((splatJson) => {
            // const currentSplatNumber = splatJson["splatNumber"];
            filter = {name: splatName};
            update = {
                commandToTrainModel: 'python ' + trainFilePathInContainer + ' -s ' + containerInputFolderPath + ' --model_path ./output/' + splatName + ' --test_iterations ' + numIterations + ' --save_iterations ' + numIterations + ' --stop_iteration ' + numIterations,
                trainCommand: 'conda init bash && echo "about to exec bash" && exec bash -c "source activate ' + condaTrainingEnvName + ' && ' + 'python ' + trainFilePathInContainer + ' -s ' + containerInputFolderPath + ' --model_path ./output/' + splatName + ' --test_iterations ' + numIterations + ' --save_iterations ' + numIterations + ' --stop_iteration ' + numIterations + '"',
                pointCloudOriginalNameFilePath: path.join(__dirname, 'public/model_output/post_train/' + splatName + "/output/" + splatName + "/point_cloud/iteration_" + numIterations + "/" + DEFAULT_POINT_CLOUD_NAME),
                pointCloudNewNameFilePath: path.join(__dirname, 'public/model_output/post_train/' + splatName + "/output/" + splatName + "/point_cloud/iteration_" + numIterations + "/" + DEFAULT_CHANGED_POINT_CLOUD_NAME)
            }
            // I don't use findOneAndUpdate inside of createNewSplat because it would update the splatNumber 
            // which we don't want to do if there already is a splat in the database
            DataModel.findOneAndUpdate(filter, update)
            .then((savedData) => {
                console.log("Refreshed data to be: ", savedData);
                resolve(savedData);
            })
            .catch((err) => {
                console.error("Error in refreshing data: ", err);
                reject();
            });
        })
    });

}

/**
 * Updates a Splat document in the database with the specified name.
 * @param {string} splatName - The name of the Splat document to update.
 * @param {object} updates - The updates to apply to the Splat document. Should be in the form { webviewerImagePath: "your_new_value" }.
 * @returns {Promise<object>} A promise that resolves to the updated Splat document if successful, otherwise rejects with an error.
 */
function updateSplat(splatName, updates) {
    return new Promise ((resolve, reject) => {
        const filter = { name: splatName };
        const options = { new: true };
        // Find the document and update it
        DataModel.findOneAndUpdate(filter, updates, options)
            .then((doc) => {
                console.log("Document updated successfully:", doc["name"]);
                // Document has been updated successfully
                resolve(doc);
            })
            .catch((err) => {
                console.error("Error occurred while updating the document:", err);
                // Handle error
                reject(null);
            });
    });
}

/**
 * Finds all Splat documents in the database.
 * @returns {Promise<Array<object>>} A promise that resolves to an array of all Splat documents if successful, otherwise rejects with an error.
 */
function findAllSplatsInDB() {
    return new Promise ((resolve, reject) => {
        DataModel.find({})
        .then((docs) => {
            if (docs != null) {
                // Uncomment if you want to print out all of splats in the database
                // console.log("All Splats: " + docs);
                resolve(docs);
            } else {
                console.log("Splat is null");
                reject(null);
            }
        })
        .catch((err) => {
            console.error('Error finding splat:', err);
            reject(null);
        });
    })
}

/**
 * Finds a Splat document in the database with the specified name.
 * @param {string} splatName - The name of the Splat document to find.
 * @returns {Promise<object|null>} A promise that resolves to the found Splat document if it exists, otherwise resolves to null. Rejects with an error if an error occurs.
 */
function findSplatInDB(splatName) {
    return new Promise ((resolve, reject) => {
        DataModel.findOne({ name: splatName })
        .then((foundSplat) => {
            resolve(foundSplat);
        })
        .catch((err) => {
            console.error('Error finding splat:', err);
            reject(null);
        });
    })
    
}

/**
 * Runs preprocessing commands for a Splat document. Immediately runs the train command once finished
 * @param {string} splatName - The name of the Splat document for which to run preprocessing commands.
 */
function runPreprocessCommand(splatName) {
    findSplatInDB(splatName).then((splatJson) => {
        status = statusTypes.PREPROCESSING_STARTED;
        console.log("STATUS_TYPE (should be PREPROCESSING_STARTED): " + status);
        
        imageName = preprocessImageName;
        mountPath = splatJson["preprocessMountPath"];
        containerCommand = preprocessCommand;
        containerFolderToSavePath = containerInputFolderPath + "."; // We need the . so that it doesn't save the public directory, but everything in the directory
        localSavePath = splatJson["postProcessedImagesFolderPath"]; // We add the splatName to the path to save 

        // console.log("preprocessing image name: " + imageName);
        // console.log("preprocessing mountPath: " + mountPath);
        // console.log("preprocessing command: " + containerCommand);
        debugger;
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
                        // status = statusTypes.PREPROCESSING_ERROR
                    } else {
                        const isRunning = data.State.Running;

                        if (isRunning) {
                            console.log('Container is already running.');
                        } else {
                            // Start the container here
                            container.start((startErr) => {
                                if (startErr) {
                                    console.error('Error starting container:', startErr);
                                    // status = statusTypes.PREPROCESSING_ERROR
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
                        status = statusTypes.PREPROCESSING_ERROR
                        return;
                    }

                    // Set the command to execute
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
                            status = statusTypes.PREPROCESSING_ERROR
                            return;
                        }

                        exec1.start(function (err, stream) {
                            if (err) {
                                console.error(err);
                                status = statusTypes.PREPROCESSING_ERROR
                                return;
                            }

                            container.modem.demuxStream(stream, process.stdout, process.stderr);

                            stream.on('end', function () {
                                console.log("SAVING");
                                refreshFolderSync(localSavePath);
                                const saveOutputCommand = 'docker cp ' + container.id + ':' + containerFolderToSavePath + ' ' + localSavePath;

                                // Save the output of the preprocess command
                                exec(saveOutputCommand, (error, stdout, stderr) => {
                                    if (error) {
                                        status = statusTypes.PREPROCESSING_ERROR
                                        console.error(`Error executing the command: ${error.message}`);
                                        return;
                                    }

                                    if (stderr) {
                                        status = statusTypes.PREPROCESSING_ERROR
                                        console.error(`Command execution produced an error: ${stderr}`);
                                        return;
                                    }

                                    // console.log(`Command output preprocess: ${stdout}`);
                                    status = statusTypes.PREPROCESSING_ENDED;
                                    console.log("STATUS_TYPE (should be PREPROCESSING_ENDED): " + status);
                                    
                                });
                                // Stop the container
                                container.stop(function (err, data) {
                                    if (err) {
                                        status = statusTypes.PREPROCESSING_ERROR
                                        console.error('Error stopping preprocessing container:', err);
                                    } else {
                                        // console.log('Preprocessing container stopped successfully:', data);
                                        runTrainCommand(splatName);
                                    }
                                    // Start training after we have preprocessed the images. If you don't want this to be
                                    // nested you could make this function into a promise and only have the runTrainCommand
                                    // function run after the promise has returned.
                                    
                                });
                                return containerId;
                            });
                        });
                    });
                });
            }
        });
    });
}

/**
 * Runs training commands for a Splat document.
 * @param {string} splatName - The name of the Splat document for which to run training commands.
 */
function runTrainCommand(splatName) {
    
    findSplatInDB(splatName).then((splatJson) => {
        status = statusTypes.TRAINING_STARTED;
        console.log("STATUS_TYPE (should be TRAINING_STARTED): " + status);

        imageName = trainingImageName;
        mountPath = splatJson["trainingMountPath"];
        containerCommand = splatJson["trainCommand"];
        containerFolderToSavePath = containerTrainedOutputPath;
        localSavePath = splatJson["trainedSplatTempFolderPath"];
        serverSavePath = splatJson["trainedSplatFolderPath"];

        // console.log("image name: " + imageName);
        // console.log("mountPath: " + mountPath);
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
                status = statusTypes.TRAINING_ERROR
                return;
            } else {
                // Get the containerId
                const containerId = container.id;

                // Start the container
                container.start(function (err, data) {
                    if (err) {
                        console.error(err);
                        status = statusTypes.TRAINING_ERROR
                        return;
                    }

                    // Set the command to execute
                    console.log("containerCommand: " + containerCommand);
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
                                status = statusTypes.TRAINING_ERROR
                                return;
                            }

                            container.modem.demuxStream(stream, process.stdout, process.stderr);

                            stream.on('end', function () {
                                console.log("SAVING");
                                // Refreshes the train folder in the model_output/post_train folder
                                refreshFolderSync(serverSavePath);
                                // Refreshes the train folder in the nerfstudio_gaussviewer folder
                                refreshFolderSync(localSavePath);

                                const saveOutputCommandOnServer = 'docker cp ' + container.id + ':' + containerFolderToSavePath + ' ' + serverSavePath;
                                const saveOutputToNerfstudioViewer =  "cp -r " + serverSavePath + " " + "public/nerfstudio_gaussviewer/nerfstudio/model_output/post_train";
                                console.log("Saving to nerf studio folder: " + saveOutputToNerfstudioViewer)
                                // We first save the output into model_output/post_train
                                exec(saveOutputCommandOnServer, (error, stdout, stderr) => {
                                    if (error) {
                                        console.error(`Error executing the command: ${error.message}`);
                                        status = statusTypes.TRAINING_ERROR
                                        return;
                                    }

                                    if (stderr) {
                                        console.error(`Command execution produced an error: ${stderr}`);
                                        status = statusTypes.TRAINING_ERROR
                                        return;
                                    }
                                    console.log(`Command output train: ${stdout}`);
                                    // TODO: Possibly delete this?
                                    // // Update the splat in the database so that we have the point cloud file path to use that is updated with the number of iterations
                                    // updateSplat(splatName, {pointCloudOriginalNameFilePath: path.join(__dirname, 'public/model_output/post_train/' + splatName + "/output/" + splatName + "/point_cloud/iteration_" + numIterations + "/" + DEFAULT_POINT_CLOUD_NAME), pointCloudNewNameFilePath: path.join(__dirname, 'public/model_output/post_train/' + splatName + "/output/" + splatName + "/point_cloud/iteration_" + numIterations + "/" + DEFAULT_CHANGED_POINT_CLOUD_NAME)})
                                    // .then((_) => {
                                        // Create the camera path that will be used to render the video
                                        createCameraPathFile(splatName)
                                        .then(() => {
                                            // Create the bounding box file in the model's frame
                                            createBoundingBoxFile(splatName)
                                                .then(()=>{
                                                    // Cull the splat points outside of the bounding box and save new .ply file in model_output/post_train folder
                                                    performBackgroundCulling(splatName)
                                                        .then(()=>{
                                                            // Save the stuff in model_output/post_train folder to the nerfstudio_gaussviewer folder
                                                            runSystemCommand(saveOutputToNerfstudioViewer)
                                                                .then(() => {
                                                                    console.log("saveOutputToNerfstudioViewer command ran");
                                                                    status = statusTypes.TRAINING_ENDED;
                                                                    
                                                                    createSplatFileForWebviewer(splatName);
                                                                })
                                                                .catch(error=>{
                                                                    console.error('Error:', error);
                                                                    status = statusTypes.TRAINING_ERROR
                                                                });
                                                        })
                                                        .catch(error=>{
                                                            console.error('Error:',error);
                                                            status = statusTypes.TRAINING_ERROR
                                                        });
                                                })
                                                .catch(error=>{
                                                    console.error('Error:', error);
                                                    status = statusTypes.TRAINING_ERROR
                                                });
                                        })
                                        .catch(error=> {
                                            console.error('Error:', error);
                                            status = statusTypes.TRAINING_ERROR
                                        });
                                    // })
                                    // .catch((err)=> {
                                    //     console.log("Updating splat in train command error: ", err)
                                    // })
                                });

                                // Stop the container
                                container.stop(function (err, data) {
                                    if (err) {
                                        console.error('Error stopping train container:', err);
                                        status = statusTypes.TRAINING_ERROR
                                    } else {
                                        console.log('Train container stopped successfully:', data);
                                        runRenderCustomPathCommand(splatName);
                                    }
                                });
                                return containerId;
                            });
                        });
                    });
                });
            }
        });
    });
}

/**
 * Runs rendering commands for a Splat document.
 * @param {string} splatName - The name of the Splat document for which to run rendering commands.
 */
function runRenderCustomPathCommand(splatName) {
    findSplatInDB(splatName)
    .then((splatJson) => {
        status = statusTypes.RENDERING_STARTED;
        const imageName = renderImageName;
        const mountPath = renderMountPath;
        const renderCommand = 'python nerfstudio/scripts/gaussian_splatting/render.py interpolate --model-path ' + splatJson["containerRenderInputFolderPath"] + ' --pose-source train --output-path ' + splatJson["renderedOutputFolderPath"] + DEFAULT_VIDEO_NAME;
        
        const containerCommand = "cd /nerfstudio_gaussviewer/nerfstudio ; pip install ./submodules/diff-gaussian-rasterization ./submodules/simple-knn ; " + renderCommand;
        const containerFolderToSavePath = renderStartingContainerPath + splatJson["renderedOutputFolderPath"] + "."; // We must add the "." so that we only save what is in the directory
        const localSavePath = splatJson["renderedOutputFolderPath"];

        // console.log("image name: " + imageName);
        // console.log("mountPath: " + mountPath);
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
                status = statusTypes.RENDERING_ERROR
                return;
            } else {
                // Get the containerId
                const containerId = container.id;

                // Start the container
                container.start(function (err, data) {
                    if (err) {
                        console.error(err);
                        status = statusTypes.RENDERING_ERROR
                        return;
                    }

                    // Set the command to execute
                    console.log("render containerCommand: " + containerCommand);
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
                            status = statusTypes.RENDERING_ERROR
                            return;
                        }

                        exec1.start(function (err, stream) {
                            if (err) {
                                console.error(err);
                                status = statusTypes.RENDERING_ERROR
                                return;
                            }

                            container.modem.demuxStream(stream, process.stdout, process.stderr);

                            stream.on('end', function () {
                                // docker cp <container_id_or_name>:<container_path> <local_path>
                                refreshFolderSync(localSavePath);
                                const saveOutputCommand = 'docker cp ' + container.id + ':' + containerFolderToSavePath + ' ' + localSavePath;

                                // Save the output of the preprocess command
                                exec(saveOutputCommand, (error, stdout, stderr) => {
                                    if (error) {
                                        console.error(`Error executing the command: ${error.message}`);
                                        status = statusTypes.RENDERING_ERROR
                                        return;
                                    }

                                    if (stderr) {
                                        console.error(`Command execution produced an error: ${stderr}`);
                                        status = statusTypes.RENDERING_ERROR
                                        return;
                                    }

                                    console.log(`Command output custom render: ${stdout}`);
                                });
                                // Stop the container
                                container.stop(function (err, data) {
                                    if (err) {
                                        console.error('Error stopping render container:', err);
                                        status = statusTypes.RENDERING_ERROR
                                    } else {
                                        console.log('Render container stopped successfully:', data);
                                    }
                                });
                                status = statusTypes.RENDERING_ENDED;
                                console.log("STATUS_TYPE (should be RENDERING_ENDED): " + status);
                                return containerId;
                            });
                        });
                    });
                    return containerId;
                });
            }
        });
    });
}
// TODO" Delete all of the console.log("STATUS_TYPE (should be RENDERING_ENDED): " + status);
                                

/**
 * Retrieves the name of the first subdirectory within a given directory.
 * @param {string} mainDirectory - The path of the directory to search for subdirectories.
 * @returns {string} The name of the first subdirectory found within the main directory.
 *                   Returns "-1" if there are either zero or multiple subdirectories.
 */
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

/**
 * Retrieves the name of the first PNG file within a given directory.
 * @param {string} folderPath - The path of the directory to search for PNG files.
 * @returns {string} The name of the first PNG file found within the folder.
 */
function getFirstPNGFileName(folderPath) {
    // Read the contents of the specified folder
    const files = fs.readdirSync(folderPath);

    // Find the first PNG file
    const pngFile = files.find(file => path.extname(file).toLowerCase() === '.png');

    return pngFile;
}

/**
 * Retrieves the names of all subdirectories within a given directory.
 * @param {string} mainDirectory - The path of the directory to search for subdirectories.
 * @returns {Array<string>} An array containing the names of all subdirectories found within the main directory.
 */
function getAllSubdirectoryNames(mainDirectory) {
    // Read the contents of folder1
    const files = fs.readdirSync(mainDirectory);

    // Filter out only directories
    const subdirectories = files.filter(file => fs.statSync(`${mainDirectory}/${file}`).isDirectory());

    return subdirectories
}

/**
 * Deletes all files and subdirectories within a given directory.
 * @param {string} directory - The path of the directory whose contents are to be deleted.
 */
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

/**
 * Creates a new directory at the specified path.
 * @param {string} directory - The path of the directory to be created.
 */
function createDirectory(directory) {
    // Create a new folder within folderA
    fs.mkdirSync(directory);
}

/**
 * Refreshes a folder by deleting its contents if it exists or creating it if it doesn't exist.
 * @param {string} folderPath - The path of the folder to be refreshed.
 */
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

/**
 * Executes a system command asynchronously using the `exec` function from the `child_process` module.
 * @param {string} command - The command to execute.
 * @returns {Promise<void>} A Promise that resolves when the command execution is successful and rejects if there is an error.
 */
function runSystemCommand(command) {
    return new Promise((resolve, reject) => {
        // Save the output of the preprocess command into nerfstudio_gaussiviewer/nerfstudio
        exec(command, (error, stdout, stderr) => {
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
            console.log(`Command output train executing rest: ${stdout}`);
            console.log("The following command was executed: " + command);
            resolve();
        });
    });
}

/**
 * Performs background culling for a given splat name asynchronously.
 * @param {string} splatName - The name of the splat to perform background culling on.
 * @returns {Promise<void>} A Promise that resolves when the background culling is successful and rejects if there is an error.
 */
function performBackgroundCulling(splatName) {
        // python ply_cull.py input.ply -b text_bbox.json -o out.ply
    return new Promise((resolve, reject) => {
        findSplatInDB(splatName).then((splatJson) => {
            const sourceFilePath = splatJson["pointCloudOriginalNameFilePath"]; 
            const destinationFilePath = splatJson["pointCloudNewNameFilePath"]; 
            console.log("sourceFilePath: " + sourceFilePath);
            console.log("destinationFilePath: " + destinationFilePath);
            console.log("About to call fs.renameSync");
            // DONT MAKE THIS fs.renameSync!! It makes the callback not work!
            fs.rename(sourceFilePath, destinationFilePath, (err) => {
                if (err) {
                    console.error('Error renaming file:', err);
                    reject();
                    return;
                }
                const originalPointCloudPath = destinationFilePath;
                const modelBoundingBoxPath = splatJson["modelBoundingBoxFilePath"] ; // path.join(__dirname, BOUNDING_BOX_FOLDER + splatName + "/" + DEFAULT_MODEL_BOUNDING_BOX_NAME);
                const outputPath = sourceFilePath ; 
                const cameraPythonScriptPath = path.join(__dirname, 'public/scripts/ply_cull.py');
                // Must make a pip virtual environment by doing "pip install virtual env" and then "virtual env python-server-commands" outside of the server
                const boundingBoxCommand = '. python-server-commands/bin/activate ; pip install open3d ; pip install plyfile ; python3 ' + cameraPythonScriptPath + ' ' + originalPointCloudPath + ' -b ' + modelBoundingBoxPath + ' -o ' + outputPath;

                console.log("Dir name: " + __dirname);
                console.log("boundingBoxFilePath name: " + originalPointCloudPath);
                console.log("outputPath name: " + outputPath);
                console.log("Camera path command: " + boundingBoxCommand);

                exec(boundingBoxCommand, (error, stdout, stderr) => {
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
                    console.log(`Command output performBoundingBoxCulling: ${stdout}`);

                    console.log('performBoundingBoxCulling function has ran');

                    resolve();
                });
                // runRenderCustomPathCommand(splatName);
                console.log('.ply uploaded and renamed successfully');
            });
        });
    });
}

/**
 * Creates a bounding box file for a given splat name asynchronously.
 * @param {string} splatName - The name of the splat to create the bounding box file for.
 * @returns {Promise<void>} A Promise that resolves when the bounding box file creation is successful and rejects if there is an error.
 */
function createBoundingBoxFile(splatName) {
    
    return new Promise((resolve, reject) => {
            
        findSplatInDB(splatName).then((splatJson) => {
            const boundingBoxInputFilePath = splatJson["appBoundingBoxFilePath"];
            const appCameraPosesFilePath = splatJson["appCameraPosesFilePath"];
            const modelCameraPosesFilePath = splatJson["modelCameraPosesFilePath"];
            const outputPath = splatJson["modelBoundingBoxFilePath"];
            const pythonScriptPath = path.join(__dirname, 'public/scripts/get_bounding_box_model_frame.py');
            // Must make a pip virtual environment by doing "pip install virtual env" and then "virtual env python-server-commands" outside of the server
            const cameraPathCommand = '. python-server-commands/bin/activate ; pip install open3d ; python3 ' + pythonScriptPath + ' ' + boundingBoxInputFilePath + ' ' + appCameraPosesFilePath + ' ' + modelCameraPosesFilePath + ' ' + outputPath;

            console.log("Dir name: " + __dirname);
            console.log("boundingBoxFilePath name: " + boundingBoxInputFilePath);
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
                console.log(`Command output createBoundingbox: ${stdout}`);

                console.log('createBoundingbox function has ran');

                resolve();
            });
            // runRenderCustomPathCommand(splatName);
            console.log('Files uploaded and renamed successfully');
        });
    });
}

/**
 * Creates a camera path file for a given splat name asynchronously.
 * @param {string} splatName - The name of the splat to create the camera path file for.
 * @returns {Promise<void>} A Promise that resolves when the camera path file creation is successful and rejects if there is an error.
 */
function createCameraPathFile(splatName) {
    
    return new Promise((resolve, reject) => {
        
        findSplatInDB(splatName).then((splatJson) => {
            const sourceFilePath = path.join(__dirname, splatJson["trainedSplatFolderPath"] + "/output/" + splatName + "/" + DEFAULT_CAMERAS_NAME);
            const destinationFilePath = path.join(__dirname, splatJson["trainedSplatFolderPath"] + "/output/" + splatName + "/" + DEFAULT_CHANGED_CAMERAS_NAME);
            console.log("sourceFilePath: " + sourceFilePath);
            console.log("destinationFilePath: " + destinationFilePath);
            console.log("About to call fs.renameSync");
            // DONT MAKE THIS fs.renameSync!! It makes the callback not work!
            fs.rename(sourceFilePath, destinationFilePath, (err) => {
                if (err) {
                    console.error('Error renaming file:', err);
                    reject();
                    return res.status(500).send('Error renaming file');
                }
                const boundingBoxFilePath = path.join(__dirname, BOUNDING_BOX_FOLDER + splatName + "/" + DEFAULT_APP_BOUNDING_BOX_NAME);
                const appCameraPosesFilePath = path.join(__dirname,  transformsFolderPath + splatName + "/" + DEFAULT_APP_CAMERA_POSE_NAME);
                const modelCameraPosesFilePath = path.join(__dirname, splatJson["trainedSplatFolderPath"] + "/output/" + splatName + "/" + DEFAULT_CHANGED_CAMERAS_NAME);
                const outputPath = path.join(__dirname, splatJson["trainedSplatFolderPath"] + "/output/" + splatName + "/" + DEFAULT_CAMERAS_NAME);
                const cameraPythonScriptPath = path.join(__dirname, 'public/scripts/get_camera_poses.py');
                // Must make a pip virtual environment by doing "pip install virtual env" and then "virtual env python-server-commands" outside of the server
                const cameraPathCommand = '. python-server-commands/bin/activate ; pip install open3d ; python3 ' + cameraPythonScriptPath + ' ' + boundingBoxFilePath + ' ' + appCameraPosesFilePath + ' ' + modelCameraPosesFilePath + ' ' + outputPath + ' 100';

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
                // runRenderCustomPathCommand(splatName);
                console.log('Files uploaded and renamed successfully');
            });
        });
    });
    
}

/**
 * Populates JSON with splat data asynchronously for a given splat name.
 * @param {string} splatName - The name of the splat to populate JSON with data for.
 */
function createSplatFileForWebviewer(splatName) {
    // First we must run convert.py
    findSplatInDB(splatName).then((splatJson) => {
        const convertToSplatCommand = ". python-server-commands/bin/activate ; pip install plyfile ; python3 public/webviewer/convert.py " + splatJson["trainedSplatFolderPath"] + "/output/" + splatName + 
                                        "/point_cloud/iteration_" + numIterations + "/point_cloud.ply --output " + splatJson["webviewerSplatFilePath"];
        console.log("Convert to splatCommand: " + convertToSplatCommand);

        const createPlaceholderFileCommand = "touch " + splatJson["webviewerSplatFilePath"];

        var imageFolderPath = "/model_output/post_processing/" + splatName + "/input/"
        var webviewerImagePath = imageFolderPath + getFirstPNGFileName("public" + imageFolderPath);// "/webviewer/images/hmc-logo.png";
        
        console.log("webviewerImagePath: " + webviewerImagePath);

        const splatUpdate = {webviewerImagePath : webviewerImagePath};
        // Thuis updateSplat has sto stay because the image path is made at runtime
        updateSplat(splatName, splatUpdate)
        .then((splatJson) => {
            refreshFolderSync(splatJson["webviewerSplatFolderPath"]);
            runSystemCommand(createPlaceholderFileCommand)
                .then(() => {
                    runSystemCommand(convertToSplatCommand)
                        .then(() => {
                            updateWebviewerInfo();
                        })
                        .catch(error=>{
                            console.error('Error:',error);
                        });
                })
                .catch(error=>{
                    console.error('Error:',error);
                });
        });
    });
}

/**
 * DEPRECATED. Adds a JSON object to a JSON file asynchronously. 
 * We no longer use this JSON file to keep track of object paths.
 * @param {Object} jsonObject - The JSON object to add.
 * @param {string} filePath - The file path of the JSON file.
 * @returns {Promise<void>} A Promise that resolves when the JSON object is successfully added to the file and rejects if there is an error.
 */
function addJsonObjectToFile(jsonObject, filePath) {
    // Read the existing JSON file
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading file:', err);
            return;
        }

        let jsonArray;
        try {
            // Parse the JSON data
            jsonArray = JSON.parse(data);
        } catch (parseError) {
            console.error('Error parsing JSON:', parseError);
            return;
        }

        // Add the new JSON object to the array
        jsonArray.push(jsonObject);

        // Convert the updated array back to JSON string
        const updatedJsonString = JSON.stringify(jsonArray, null, 2);

        // Write the updated JSON data back to the file
        fs.writeFile(filePath, updatedJsonString, 'utf8', (writeErr) => {
            if (writeErr) {
                console.error('Error writing file:', writeErr);
                return;
            }
            console.log('JSON object added successfully!');
        });
    });
}

/**
 * Retrieves the number of splats in a JSON asynchronously.
 * @returns {Promise<number>} A Promise that resolves with the number of splats in the JSON if successful, otherwise rejects with an error.
 */
function getNumSplatsInJson() {
    return new Promise((resolve, reject) => {
        findAllSplatsInDB()
        .then((docs) => {
            console.log("Num splats: ", docs.length);
            resolve(docs.length);
        }).catch((err) => {
            console.error("Error in getting number of splats: ", err);
            reject(-1);
        });
    });
    // return new Promise((resolve, reject) => {
    //     fs.readFile(webviewerSplatJson, 'utf8', (err, data) => {
    //         if (err) {
    //             console.error(err);
    //             resolve(-1);
    //         }

    //         try {
    //             const jsonArray = JSON.parse(data);
    //             const count = jsonArray.length;
    //             resolve(count)
    //         } catch (error) {
    //             console.error(error);
    //             reject(-1);
    //         }
    //     });
    // });
}

 /**
 * Updates web viewer information asynchronously.
 * @returns {Promise<void>} A Promise that resolves when the web viewer information is successfully updated and rejects if there is an error.
 */
 function updateWebviewerInfo() {
    return new Promise((resolve, reject) => {
        if (usingMongo) {
            var template = fs.readFileSync(path.join(__dirname, 'views', 'index.ejs'), 'utf-8');
            findAllSplatsInDB()
            .then((splats) => {
                splats.forEach(obj => {
                    app.get(`/object/${obj._id}`, (req, res) => {
                        const htmlContent = template.replace('{{title}}', obj.name)
                                                    .replace('{{splatPath}}', obj.webviewerSplatFilePath)
                                                    .replace('{{modelBoundingBoxPath}}', obj.modelBoundingBoxRelativeFilePath);
                        res.send(htmlContent);
                    });
                });
            })
            .catch((err)=> {
                console.log("Error creating get page for all splats: ", err);
            });
            resolve();
        } else {
            
                // JSON file with data on rendered splats
                objectData = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'objects.json'), 'utf-8'));
        
                // Read HTML template (antimatter webviewer)
                template = fs.readFileSync(path.join(__dirname, 'views', 'index.ejs'), 'utf-8');
                // Route for each object page
                objectData.forEach(obj => {
                    app.get(`/object/${obj.id}`, (req, res) => {
                        const htmlContent = template.replace('{{title}}', obj.title)
                                                    .replace('{{imagePath}}', obj.imagePath)
                                                    .replace('{{splatPath}}', obj.splatPath)
                                                    .replace('{{videoPath}}', obj.videoPath)
                                                    .replace('{{modelBoundingBoxPath}}', obj.modelBoundingBoxRelativeFilePath);;
                        res.send(htmlContent);
                    });
                });
                resolve();
        }
    });
    
}

const PORT = 15002;
//process.env.PORT || 3000;
const galleryTemplate = fs.readFileSync(path.join(__dirname, 'views', 'gallery.ejs'), 'utf-8');

app.use(express.static("public"))
app.set('view engine', 'ejs');

 // ------ WebViewer ------
 if (usingMongo) {
    // Used to make get requests for each of the splats in the mongo database
    var template = fs.readFileSync(path.join(__dirname, 'views', 'index.ejs'), 'utf-8');
    findAllSplatsInDB()
    .then((splats) => {
        splats.forEach(obj => {
            app.get(`/object/${obj._id}`, (req, res) => {
                const htmlContent = template.replace('{{title}}', obj.name)
                                            .replace('{{splatPath}}', obj.webviewerSplatFilePath)
                                            .replace('{{modelBoundingBoxPath}}', obj.modelBoundingBoxRelativeFilePath);
                res.send(htmlContent);
            });
        });
    })
    .catch((err)=> {
        console.log("Error creating get page for all splats: ", err);
    })
 } else {
    // JSON file with data on rendered splats
   var objectData = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'objects.json'), 'utf-8'));
   
   // Read HTML template (antimatter webviewer)
   var template = fs.readFileSync(path.join(__dirname, 'views', 'index.ejs'), 'utf-8');
   // Route for each object page
   objectData.forEach(obj => {
       app.get(`/object/${obj.id}`, (req, res) => {
           const htmlContent = template.replace('{{title}}', obj.title)
                                       .replace('{{splatPath}}', obj.splatPath)
                                       .replace('{{modelBoundingBoxPath}}', obj.boundingBoxPath);
           res.send(htmlContent);
       });
   });
 }

 /**
 * Allows the user to upload a .zip file from the ipad app. Displays all the splats in the database
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
 app.get('/', (req, res) => {
    console.log("HOME SCREEN");
    // On start we clear the database and re-populate so that we have everything in the database 
    // that is in the server. THIS ISN'T GOOD IN THE LONG TERM. We only do this because there 
    // are splats in the server that aren't in the database because they were made before the database
    // was a thing
    // if (onStart == 1) {
    //     deleteAllSplatsInDB()
    //     .then(() => {
    //         const renderedSplats = getAllSubdirectoryNames(videosDirectory + '/');
    //         console.log('renderedSplats: ', renderedSplats);
    //         createMultipleSplatsInDB(renderedSplats)
    //         .then(() => {
    //             splatNameList = getAllSubdirectoryNames(imageFolderPath);
    //             res.render("upload", { splatNameList: splatNameList });
    //         });
    //         onStart++;
    //     });
    // }
    // } else {
    splatNameList = getAllSubdirectoryNames(imageFolderPath);
    res.render("upload", { splatNameList: splatNameList });
    // }
    // refreshFolderSync("public/model_output/") // I use this to clear model_output folder when testing
    // createNewSplatDataOnDB(splatName)
    // .then(() => {
    //     splatNameList = getAllSubdirectoryNames(imageFolderPath);
    //     res.render("upload", { splatNameList: splatNameList });
    // });
});

/**
 * Used for testing stuff
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
app.get('/temp_home', (req, res) => {
    // refreshFolderSync("public/model_output/") // I use this to clear model_output folder when testing
    splatNameList = getAllSubdirectoryNames(imageFolderPath);
    res.render("upload", { splatNameList: splatNameList })
});

/**
 * Used to let the ipad download the splat
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
app.get('/download', (req, res) => {
    const splatPath = req.query.splatPath; // Get splatPath from the query string
    const filePath = path.join(__dirname, splatPath);
    res.setHeader('Content-Disposition', `attachment; filename=${path.basename(splatPath)}`);
    res.sendFile(filePath);
})

/**
 * Lets users see the webviewer front page.
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
app.get('/webviewer', (req, res) => {
    updateWebviewerInfo()
    .then(() => {
        if (usingMongo) {
            findAllSplatsInDB().then((docs) => {
                const galleryItems = docs.map(obj =>
                    `<div class="item">
                        <h2 class="item-title">${obj.name}</h2>
                        <a href="/object/${obj._id}">Closer Look</a>
                        <div class="media-container">
                            <img src="${obj.webviewerImagePath}" class="preview-image"></img>
                            <video class="video" loop muted>
                                <source src="${obj.webviewerVideoPath}" type="video/mp4">
                                Your browser does not support the video tag
                            </video>
                        </div>
                    </div>`
                ).join('');
                // Insert gallery items into gallery template
                const galleryContent = galleryTemplate.replace('{{galleryItems}}', galleryItems);
                res.send(galleryContent);
            });
        }
        else {
            // Generate gallery items for each object
            const galleryItems = objectData.map(obj =>
                `<div class="item">
                    <h2 class="item-title">${obj.title}</h2>
                    <a href="/object/${obj.id}">Closer Look</a>
                    <div class="media-container">
                        <img src="${obj.imagePath}" class="preview-image"></img>
                        <video class="video" loop muted>
                            <source src="${obj.videoPath}" type="video/mp4">
                            Your browser does not support the video tag
                        </video>
                    </div>
                </div>`
            ).join('');
            // Insert gallery items into gallery template
            const galleryContent = galleryTemplate.replace('{{galleryItems}}', galleryItems);
            res.send(galleryContent);
        }
    })
    .catch(error=>{
        console.error('Error:',error);
    });
    
});

/**
 * Gets the webviewer link from the splat that was requested. Used to let
 * the iPad user look at splats
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
app.get('/get_webviewer_link/:splatName', (req, res) => {
    const splatName = req.params.splatName;
    findSplatInDB(splatName)
    .then((splatJson) => {
        if (splatJson != null) {
            const webviewerLink = splatJson["webviewerLink"];
            console.log("Webviewer LINK: ", webviewerLink);
            status = statusTypes.WAITING_FOR_DATA;
            res.send({webviewerLink: webviewerLink});
        } else {
            res.status(404).send("Splat not found");
        }
    })
    .catch((err) => {
        console.log("Error finding splat in database: " + err);
    });
});

/**
 * Route to handle retrieving splat videos
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
app.get('/download_video/:splatName', (req, res) => {
    const splatName = req.params.splatName;
    const filePath = path.join(__dirname, videosDirectory, splatName, DEFAULT_VIDEO_NAME);
    console.log("File path to download:" + filePath);
    // Check if the file exists
    if (fs.existsSync(filePath)) {
        // Set the appropriate headers for the response
        res.setHeader('Content-disposition', 'attachment; filename=' + splatName + ".mp4");
        res.setHeader('Content-type', 'video/mp4');

        // Stream the file to the client
        const fileStream = fs.createReadStream(filePath);
        status = statusTypes.WAITING_FOR_DATA
        fileStream.pipe(res);
    } else {
        // If the file does not exist, send a 404 error
        res.status(404).send('File not found');
    }
});

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



/**
 * Route to handle uploads from the iPad
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
app.post('/upload_device_data', tmp_upload.array('files'), (req, res) => {
    status = statusTypes.DATA_UPLOAD_STARTED;

    splatName = req.body.splatName; // The folder name specified in the request body
    console.log("Splat name from ipad: " + splatName);
    
    // Assuming a single ZIP file upload for simplicity; adjust as necessary for multiple files
    if (req.files.length > 0) {
        const zipFilePath = req.files[0].path; // gives full path
        const extractDir = path.dirname(zipFilePath);
        console.log(extractDir)
        if (!fs.existsSync(extractDir)){
            fs.mkdirSync(extractDir, { recursive: true });
        }

        var didError = false;
        // Unzip the file
        try {
            // Unzip the file into the temp extraction directory
            const zip = new admZip(zipFilePath);
            zip.extractAllTo(extractDir, true); // true for overwrite
            console.log(`Successfully extracted ${req.files[0].originalname} to ${extractDir}`);
            const originalName = req.files[0].originalname.replace(/\.zip$/i, '');
            const extractedFile = path.join(extractDir, originalName);
            console.log(extractedFile);
            // move bounding box
            const boundingbox_path = path.join(extractedFile, DEFAULT_APP_BOUNDING_BOX_NAME);
            const final_boundingbox_folder = path.join(__dirname, BOUNDING_BOX_FOLDER, splatName);
            refreshFolderSync(final_boundingbox_folder);
            if (!fs.existsSync(final_boundingbox_folder)){
                fs.mkdirSync(final_boundingbox_folder, { recursive: true });
            }
            fs.rename(boundingbox_path, path.join(final_boundingbox_folder, DEFAULT_APP_BOUNDING_BOX_NAME), (err) => {
                if (err) throw err;
                console.log(`Moved ${boundingbox_path} to ${path.join(final_boundingbox_folder, DEFAULT_APP_BOUNDING_BOX_NAME)}`);
            })

            // move transforms.json same as above
            const transforms_path = path.join(extractedFile, 'transforms.json')
            const final_transforms_folder = path.join(__dirname, transformsFolderPath, splatName);
            refreshFolderSync(final_transforms_folder);
            if (!fs.existsSync(final_transforms_folder)){
                fs.mkdirSync(final_transforms_folder, { recursive: true });
            }
            fs.rename(transforms_path, path.join(final_transforms_folder, DEFAULT_APP_CAMERA_POSE_NAME), (err) => {
                if (err) throw err;
                console.log(`Moved ${transforms_path} to ${path.join(final_transforms_folder, DEFAULT_APP_CAMERA_POSE_NAME)}`);
            })

            // move depth and rgb images
            const image_uploads_path = path.join(extractedFile, 'images')
            const final_image_uploads_folder = path.join(__dirname, imageFolderPath, splatName, 'input'); 
            const final_depth_uploads_folder = path.join(__dirname, depthFolderPath, splatName, 'input'); 
            refreshFolderSync(final_image_uploads_folder);
            refreshFolderSync(final_depth_uploads_folder);

            if (!fs.existsSync(final_image_uploads_folder)){
                fs.mkdirSync(final_image_uploads_folder, { recursive: true });
            }
            if (!fs.existsSync(final_depth_uploads_folder)){
                fs.mkdirSync(final_depth_uploads_folder, { recursive: true });
            }
            const files = fs.readdirSync(image_uploads_path);
            for (const file of files) {
                    const full_image_path = path.join(image_uploads_path, file)
                    const final_full_image_path = path.join(final_image_uploads_folder, file)
                    console.log(final_full_image_path);
                    const final_full_depth_path = path.join(final_depth_uploads_folder, file)
                    if (file.includes('.depth.png')) {
                        fs.rename(full_image_path, final_full_depth_path, (err) => {
                            if (err) throw err;
                            console.log(`Moved ${full_image_path} to ${final_full_depth_path}`);
                        })
                    } else if (file.includes('.png')) { 
                        fs.rename(full_image_path, final_full_image_path, (err) => {
                            if (err) throw err;
                            console.log(`Moved ${full_image_path} to ${final_full_image_path}`);
                        })
                    }
            }
            status = statusTypes.DATA_UPLOAD_ENDED;
            console.log("STATUS_TYPE (should be DATA_UPLOAD_ENDED): " + status);
            res.json({'message': 'ZIP file processed and files distributed successfully'});
        
        
        
        } catch (error) {
            didError = true;
            status = statusTypes.WAITING_FOR_DATA;
            console.error(`Error extracting ZIP file: ${error}`);
            res.status(500).send('Failed to extract ZIP file');
        } finally {
            // We only want to create the splat on the database after all the 
            // files have been uploaded and if we did not error
            if (!didError) {
                createNewSplatDataOnDB(splatName)
                .then((_) => {
                    console.log('starting full pipeline');
                    runPreprocessCommand(splatName);
                })
                .catch((err) => {
                    console.log("Error in creating splat in upload_device_data", err);
                });
            }
        }
    } else {
        status = statusTypes.WAITING_FOR_DATA;
        res.status(400).send('No files uploaded');
    }
    
});

/**
 * DEPRECATED. This is only used to test current splats in the database
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
app.post('/upload_splat_name', (req, res) => {
    splatName = req.body.splatName;
    res.send("Splat name recorded as: " + splatName);
});

/**
 * DEPRECATED. Only used for testing.
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
// app.post('/full_pipeline', (req, res) => {
//     // The preprocess command starts the pieline
//     runPreprocessCommand(splatName);
//     res.send("Pipeline Starting");
// });

/**
 * DEPRECATED. Only used for testing.
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
// app.post('/preprocess', (req, res) => {
//     // status = statusTypes.PREPROCESSING_STARTED;
//     runPreprocessCommand(splatName);
//     // status = statusTypes.PREPROCESSING_ENDED;
//     res.send("Images Processing Attempted");
// });

/**
 * DEPRECATED. Only used for testing.
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
app.post('/train', (req, res) => {
    runTrainCommand(splatName); 
    res.send("Model Training Attempted");

});

/**
 * Allows user to reset the server status
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
app.post('/reset_status', (req, res) => {
    status = statusTypes.WAITING_FOR_DATA;
    res.send({"status": status});
})

/**
 * Allows user to query for the server status
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
app.get('/status', (req, res) => {
    res.send({"status": status});
})

/**
 * DEPRECATED. Only used for testing.
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
// app.post('/render_custom_path', (req, res) => {
//     // status = statusTypes.RENDERING_STARTED;
//     runRenderCustomPathCommand(splatName);
//     // status = statusTypes.RENDERING_ENDED;
//     res.send("Model Custom Render Attempted");
// });

/**
 * Used for testing random things. 
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
app.get('/heartbeat', (req, res) => {
    res.json({ message: "Hello World" });
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT}`);
});