Add-Type -AssemblyName System.Drawing

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$targetRoot = Join-Path $projectRoot 'frontend-second\.qa\screenshots\light\desktop-1440x900'
$referenceRoot = Join-Path $projectRoot 'OpenAgent_HTML_Reference\references'
$outputPath = Join-Path $projectRoot 'frontend-second\.qa\visual-comparison-10-approved.png'
$routes = @('home', 'chat', 'agents', 'research', 'files', 'automations', 'calendar', 'models', 'settings', 'codeq')
$labels = @('Home', 'Chat', 'Agents', 'Research', 'Files', 'Automations', 'Calendar', 'Models', 'Settings', 'CodeQ')
$sheetWidth = 1000
$headerHeight = 34
$previewWidth = 480
$previewHeight = 300
$rowHeight = $headerHeight + $previewHeight + 24
$sheetHeight = 16 + ($routes.Count * $rowHeight)
$sheet = [System.Drawing.Bitmap]::new($sheetWidth, $sheetHeight)
$graphics = [System.Drawing.Graphics]::FromImage($sheet)
$graphics.Clear([System.Drawing.Color]::FromArgb(239, 242, 247))
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAlias
$font = [System.Drawing.Font]::new('Segoe UI', 12, [System.Drawing.FontStyle]::Bold)
$brush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(31, 41, 55))

for ($index = 0; $index -lt $routes.Count; $index++) {
  $route = $routes[$index]
  $referenceNumber = '{0:D2}' -f ($index + 1)
  $y = 16 + ($index * $rowHeight)
  $target = [System.Drawing.Image]::FromFile((Join-Path $targetRoot "$route.png"))
  $reference = [System.Drawing.Image]::FromFile((Join-Path $referenceRoot "${referenceNumber}_$route.png"))
  $graphics.DrawString("$($labels[$index]) · frontend-second", $font, $brush, 10, $y)
  $graphics.DrawString("$($labels[$index]) · approved reference", $font, $brush, 510, $y)
  $graphics.DrawImage($target, 10, ($y + $headerHeight), $previewWidth, $previewHeight)
  $graphics.DrawImage($reference, 510, ($y + $headerHeight), $previewWidth, $previewHeight)
  $target.Dispose()
  $reference.Dispose()
}

$sheet.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$brush.Dispose()
$font.Dispose()
$graphics.Dispose()
$sheet.Dispose()
Write-Output $outputPath
