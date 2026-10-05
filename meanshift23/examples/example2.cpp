#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
int main(int argc, char* argv[]) {
    // Check if filenames were provided as arguments
    std::string csvFilePath = "data/input/my_data.csv";
    std::string outFilePath = "data/output/results.csv";
    
    if (argc > 1) {
        csvFilePath = argv[1];
    }
    if (argc > 2) {
        outFilePath = argv[2];
    }
    
    meanshift::Dataset dataset;
    
    std::cout << "Loading data from external CSV file: " << csvFilePath << std::endl;
    
    // Attempt to load the external CSV data
    // Assuming the file has a header row. If not, change 'true' to 'false'.
    if (!dataset.loadFromCSV(csvFilePath, true)) {
        std::cerr << "Error: Failed to load external data from " << csvFilePath << std::endl;
        std::cerr << "Please ensure the file exists and is in CSV format." << std::endl;
        return 1;
    }
    
    std::cout << "Successfully loaded external data." << std::endl;
    
    // Optional: standardize the data
    // dataset.standardize();

    std::cout << "Running MeanShift clustering on the external dataset..." << std::endl;
    
    // Set up MeanShift algorithm
    // (bandwidth, tolerance, max_iterations)
    meanshift::MeanShift ms(2.0, 1e-3, 300);
    
    // Pass the real external data
    ms.fit(dataset.getData());
    
    auto clusters = ms.getClusters();
    std::cout << "\nNumber of clusters found: " << clusters.size() << std::endl;
    
    int i = 0;
    for (const auto& c : clusters) {
        std::cout << "Cluster " << i++ << " centroid: (";
        for (std::size_t d = 0; d < c.centroid.size(); ++d) {
            std::cout << c.centroid[d] << (d == c.centroid.size() - 1 ? "" : ", ");
        }
        std::cout << ") with " << c.pointIndices.size() << " points." << std::endl;
    }
    
    std::cout << "Clustering evaluation score: " << ms.evaluate(dataset.getData()) << std::endl;
    
    // --- NEW: WRITE OUTPUT TO FILE ---
    std::cout << "\nWriting results to: " << outFilePath << " ..." << std::endl;
    
    std::ofstream outFile(outFilePath);
    if (outFile.is_open()) {
        // Write header
        outFile << "Feature1,Feature2,Cluster_ID\n";
        
        const auto& realData = dataset.getData();
        const auto& labels = ms.getLabels();
        
        for (std::size_t k = 0; k < realData.size(); ++k) {
            // Write each feature separated by a comma
            for (std::size_t d = 0; d < realData[k].size(); ++d) {
                outFile << realData[k][d] << ",";
            }
            // Write the cluster label it belongs to
            outFile << labels[k] << "\n";
        }
        outFile.close();
        std::cout << "Successfully saved!" << std::endl;
    } else {
        std::cerr << "Warning: Could not open output file for writing!" << std::endl;
    }

    return 0;
}
