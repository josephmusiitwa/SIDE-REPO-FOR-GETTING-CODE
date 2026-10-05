$dirs = @("c:\Users\josep\OneDrive\Desktop\SIDE-REPO-FOR-GETTING-CODE\meanshift23\src", "c:\Users\josep\OneDrive\Desktop\SIDE-REPO-FOR-GETTING-CODE\meanshift23\examples")

foreach ($dir in $dirs) {
    $files = Get-ChildItem -Path $dir -Filter "*.cpp"
    foreach ($file in $files) {
        $content = Get-Content $file.FullName -Raw
        
        # Avoid multiple inserts
        if ($content -notmatch 'using namespace std;') {
            # Find the last #include
            $lines = $content -split "`r`n|`n"
            $lastIncludeIdx = -1
            for ($i = 0; $i -lt $lines.Length; $i++) {
                if ($lines[$i] -match '^#include') {
                    $lastIncludeIdx = $i
                }
            }
            
            if ($lastIncludeIdx -ne -1) {
                $lines = $lines[0..$lastIncludeIdx] + "`nusing namespace std;" + $lines[($lastIncludeIdx+1)..($lines.Length-1)]
                $content = $lines -join "`n"
            } else {
                $content = "using namespace std;`n" + $content
            }
        }
        
        $content = $content -replace 'std::', ''
        
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-Host "Updated $($file.FullName)"
    }
}
