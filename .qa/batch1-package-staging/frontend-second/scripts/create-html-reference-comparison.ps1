Add-Type -AssemblyName System.Drawing

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$targetMatrix = Join-Path $projectRoot 'frontend-second\.qa\react-parity-current'
$referenceMatrix = Join-Path $projectRoot 'frontend-second\.qa\approved-html-current'
$targetManifestPath = Join-Path $targetMatrix 'manifest.json'
$referenceManifestPath = Join-Path $referenceMatrix 'manifest.json'
$canonicalHash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1'
if (!(Test-Path $targetManifestPath) -or !(Test-Path $referenceManifestPath)) { throw 'Missing current canonical capture manifest.' }
$targetManifest = Get-Content -Raw $targetManifestPath | ConvertFrom-Json
$referenceManifest = Get-Content -Raw $referenceManifestPath | ConvertFrom-Json
if ($targetManifest.sourceSha256 -ne $canonicalHash -or $referenceManifest.sourceSha256 -ne $canonicalHash) { throw 'Refusing comparison sheet: both matrices must use the pinned approved frontend.' }
$targetRoot = Join-Path $targetMatrix 'light\desktop-1440x900'
$referenceRoot = Join-Path $referenceMatrix 'light\desktop-1440x900'
$outputPath = Join-Path $projectRoot 'frontend-second\.qa\visual-match-current-comparison-sheet.png'
$routes = @('home', 'chat', 'codeq', 'agents', 'research', 'files', 'automations', 'calendar', 'models', 'knowledge', 'voice', 'assistant', 'communications', 'security', 'developer', 'engine', 'science', 'twin', 'vault', 'scraper', 'finetune', 'settings', 'tools')
$labels = @('Home', 'Chat', 'CodeQ', 'Agents', 'Research', 'Files', 'Automations', 'Calendar', 'Models', 'Knowledge', 'Voice', 'Assistant', 'Communications', 'Security', 'Developer', 'Adaptive engine', 'Scientific AI', 'Digital twin', 'Vault', 'Web scraper', 'Fine-tuning', 'Settings', 'All tools')
$cellWidth = 700
$cellHeight = 438
$headerHeight = 31
$rowHeight = $cellHeight + $headerHeight + 14
$sheet = [System.Drawing.Bitmap]::new(($cellWidth * 2 + 24), ($rowHeight * $routes.Count + 20))
$graphics = [System.Drawing.Graphics]::FromImage($sheet)
$graphics.Clear([System.Drawing.Color]::FromArgb(242, 243, 245))
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$font = [System.Drawing.Font]::new('Segoe UI', 12, [System.Drawing.FontStyle]::Bold)
$brush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(31, 41, 55))
$records = @()

for ($index = 0; $index -lt $routes.Count; $index++) {
  $route = $routes[$index]
  $y = 10 + ($index * $rowHeight)
  $targetPath = Join-Path $targetRoot "$route.png"
  $referencePath = Join-Path $referenceRoot "$route.png"
  if (!(Test-Path $targetPath)) { throw "Missing frontend-second capture: $targetPath" }
  if (!(Test-Path $referencePath)) { throw "Missing approved HTML capture: $referencePath" }
  $target = [System.Drawing.Image]::FromFile($targetPath)
  $reference = [System.Drawing.Image]::FromFile($referencePath)
  $graphics.DrawString("$($labels[$index]) · frontend-second (demo initial state)", $font, $brush, 10, $y)
  $graphics.DrawString("$($labels[$index]) · canonical approved HTML", $font, $brush, ($cellWidth + 22), $y)
  $graphics.DrawImage($target, 10, ($y + $headerHeight), $cellWidth, $cellHeight)
  $graphics.DrawImage($reference, ($cellWidth + 22), ($y + $headerHeight), $cellWidth, $cellHeight)
  $records += [PSCustomObject]@{ route = $route; target = $targetPath; reference = $referencePath; targetSize = "$($target.Width)x$($target.Height)"; referenceSize = "$($reference.Width)x$($reference.Height)" }
  $target.Dispose()
  $reference.Dispose()
}

$sheet.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$brush.Dispose()
$font.Dispose()
$graphics.Dispose()
$sheet.Dispose()
$records | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 (Join-Path (Split-Path $outputPath) 'html-reference-comparison.json')
Write-Output $outputPath
