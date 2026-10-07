#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>

using namespace std;
int main() {
    // A list of different files to process
    vector<string> inputFiles = {
        "data/input/my_data.csv",
        "data/input/biology_data.csv",
        "data/input/finance_data.csv"
    };

    cout << "--- Batch Processing Multiple Files ---" << endl;

    // Loop through each file in the list
    for (size_t i = 0; i < inputFiles.size(); ++i) {
        string currentInputFile = inputFiles[i];
        
        // Generate an output file name for each one
        string currentOutputFile = "data/output/results_file_" + to_string(i + 1) + ".csv";
        
        cout << "\n========================================" << endl;
        cout << "Processing File " << (i + 1) << " of " << inputFiles.size() << endl;
        cout << "Input: " << currentInputFile << endl;
        
        meanshift::Dataset dataset;
        if (!dataset.loadFromCSV(currentInputFile, true)) {
            cerr << "-> Skipping... (File not found or unreadable)" << endl;
            continue; // Skip to the next file in the loop
        }
        
        cout << "-> Running MeanShift clustering..." << endl;
        meanshift::MeanShift ms(2.0, 1e-3, 300);
        ms.fit(dataset.getData());
        
        //number of clusters found
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
        } else {
            cerr << "-> Warning: Could not open output file for writing!" << endl;
        }
    }
    
    cout << "\n========================================" << endl;
    cout << "All files processed successfully!" << endl;

    return 0;
}
