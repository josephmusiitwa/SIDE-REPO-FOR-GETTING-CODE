#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <filesystem>
#include <fstream>

using namespace std;
namespace fs = filesystem;

int main() {
    string inputFolder = "data/input";
    string outputFolder = "data/output";
    
    cout << "--- Auto-Scanning Batch Processor ---" << endl;
    cout << "Scanning folder: " << inputFolder << " for .csv files..." << endl;

    // Check if the input directory actually exists
    if (!fs::exists(inputFolder) || !fs::is_directory(inputFolder)) {
        cerr << "Error: Directory '" << inputFolder << "' does not exist." << endl;
        return 1;
    }

    int filesProcessed = 0;

    // Magically loop through every single file inside the input folder
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        
        // We only care about normal files that end in ".csv"
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            
            string currentInputFile = entry.path().string();
            string baseFileName = entry.path().stem().string(); // grabs just the name (e.g., "my_data")
            
            // Automatically generate a matching output file name!
            string currentOutputFile = outputFolder + "/results_" + baseFileName + ".csv";
            
            cout << "\n========================================" << endl;
            cout << "Processing File: " << baseFileName << ".csv" << endl;
            
            meanshift::Dataset dataset;
            if (!dataset.loadFromCSV(currentInputFile, true)) {
                cerr << "-> Skipping... (Failed to read CSV)" << endl;
                continue; 
            }
            
            cout << "-> Running MeanShift clustering..." << endl;
            meanshift::MeanShift ms(2.0, 1e-3, 300);
            ms.fit(dataset.getData());
            
            cout << "-> Number of clusters found: " << ms.getClusters().size() << endl;
            cout << "-> Saving to: " << currentOutputFile << endl;
            
            ofstream outFile(currentOutputFile);
            if (outFile.is_open()) {
                outFile << "Feature1,Feature2,Cluster_ID\n";
                const auto& realData = dataset.getData();
                const auto& labels = ms.getLabels();
                
                for (size_t k = 0; k < realData.size(); ++k) {
                    for (size_t d = 0; d < realData[k].size(); ++d) {
                        outFile << realData[k][d] << ",";
                    }
                    outFile << labels[k] << "\n";
                }
                outFile.close();
                cout << "-> Success!" << endl;
                filesProcessed++;
            } else {
                cerr << "-> Warning: Could not open output file for writing!" << endl;
            }
        }
    }
    
    cout << "\n========================================" << endl;
    cout << "Finished! " << filesProcessed << " files were successfully processed." << endl;

    return 0;
}
