#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
int main() {
    // A list of different files to process
    std::vector<std::string> inputFiles = {
        "data/input/my_data.csv",
        "data/input/biology_data.csv",
        "data/input/finance_data.csv"
    };

    std::cout << "--- Batch Processing Multiple Files ---" << std::endl;

    // Loop through each file in the list
    for (std::size_t i = 0; i < inputFiles.size(); ++i) {
        std::string currentInputFile = inputFiles[i];
        
        // Generate an output file name for each one
        std::string currentOutputFile = "data/output/results_file_" + std::to_string(i + 1) + ".csv";
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Processing File " << (i + 1) << " of " << inputFiles.size() << std::endl;
        std::cout << "Input: " << currentInputFile << std::endl;
        
        meanshift::Dataset dataset;
        if (!dataset.loadFromCSV(currentInputFile, true)) {
            std::cerr << "-> Skipping... (File not found or unreadable)" << std::endl;
            continue; // Skip to the next file in the loop
        }
        
        std::cout << "-> Running MeanShift clustering..." << std::endl;
        meanshift::MeanShift ms(2.0, 1e-3, 300);
        ms.fit(dataset.getData());
        
        std::cout << "-> Number of clusters found: " << ms.getClusters().size() << std::endl;
        std::cout << "-> Saving to: " << currentOutputFile << std::endl;
        
        std::ofstream outFile(currentOutputFile);
        if (outFile.is_open()) {
            outFile << "Feature1,Feature2,Cluster_ID\n";
            const auto& realData = dataset.getData();
            const auto& labels = ms.getLabels();
            
            for (std::size_t k = 0; k < realData.size(); ++k) {
                for (std::size_t d = 0; d < realData[k].size(); ++d) {
                    outFile << realData[k][d] << ",";
                }
                outFile << labels[k] << "\n";
            }
            outFile.close();
            std::cout << "-> Success!" << std::endl;
        } else {
            std::cerr << "-> Warning: Could not open output file for writing!" << std::endl;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "All files processed successfully!" << std::endl;

    return 0;
}
