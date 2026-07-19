# stage-trocr-space.ps1
#
# Stages a ready-to-push Hugging Face Space repo for the Javanese TrOCR
# fine-tune pipeline. Copies the Space SDK files, the shared Python pipeline,
# your Javanese Aksara fonts, and any scanned PDFs into ./space_repo/ and
# initializes it as a git repo pointed at HF.
#
# The Space repo is its own HF git repository - your aksara.ts working files
# (gitignored under training/fonts/, training/pdfs/) are copied in, so nothing
# leaks back to GitHub.
#
# USAGE
#   .\scripts\stage-trocr-space.ps1                           # space_repo/ only
#   .\scripts\stage-trocr-space.ps1 -InitGit                  # also git init + hint
#   .\scripts\stage-trocr-space.ps1 -InitGit `
#       -SpaceName trocr-javanese-synthetic `
#       -HfUsername yourname                                  # wires `origin` for you
#
# Then push from ./space_repo:
#   cd space_repo
#   git add -A
#   git commit -m "Initial TrOCR Space (synthetic-only)"
#   git push -u origin main --force
# (--force needed once if the Space was auto-initialized on HF with a README;
#  your staged commit replaces that stub. Harmless after that.)
#
# REQUIRES
#   git + git-lfs (https://git-lfs.github.com/). HF Hub rejects any
#   individual file >10 MiB and any binary file type it expects via LFS
#   (fonts, pdfs, images, .safetensors). With -InitGit this script writes
#   .gitattributes tracking fonts/* and pdfs/* via LFS, so the initial push
#   succeeds without manual fixup.
#   Model artifacts (.safetensors, .pth) generated inside the Space during
#   training are handled by huggingface_hub's push_to_hub automatically.

[CmdletBinding()]
param(
    [switch]$InitGit,
    [string]$SpaceName   = "trocr-javanese-synthetic",
    [string]$HfUsername  = "",
    [string]$SpaceRepoRoot = (Join-Path (Join-Path $PSScriptRoot "..") "space_repo")
)

$ErrorActionPreference = "Stop"
$SpaceRepoRoot = (Resolve-Path (New-Item -ItemType Directory -Force -Path $SpaceRepoRoot)).Path
Write-Host "[1/5] Space repo target: $SpaceRepoRoot"

# ---------------------------------------------------------------------------
# 1. Resolve source paths
# ---------------------------------------------------------------------------
$aksaraRoot     = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$trocrDir       = Join-Path (Join-Path $aksaraRoot "training") "trocr"
$spaceDir       = Join-Path $trocrDir "space"
$fontsDir       = Join-Path (Join-Path $aksaraRoot "training") "fonts"
$pdfsDir        = Join-Path (Join-Path $aksaraRoot "training") "pdfs"

foreach ($p in @($trocrDir, $spaceDir, $fontsDir)) {
    if (-not (Test-Path -LiteralPath $p)) {
        throw "Missing required source dir: $p"
    }
}

# ---------------------------------------------------------------------------
# 2. Copy Space SDK files (Dockerfile, app.py, README.md, requirements.txt)
# ---------------------------------------------------------------------------
Write-Host "[2/5] Mirroring Space SDK files …"
$spaceSdkFiles = @("Dockerfile", "app.py", "README.md", "requirements.txt")
foreach ($f in $spaceSdkFiles) {
    $src = Join-Path $spaceDir $f
    if (-not (Test-Path -LiteralPath $src)) {
        throw "Missing Space SDK file: $src"
    }
    Copy-Item -LiteralPath $src -Destination $SpaceRepoRoot -Force
}

# ---------------------------------------------------------------------------
# 3. Copy shared Python pipeline scripts (Dockerfile COPYs these)
# ---------------------------------------------------------------------------
Write-Host "[3/5] Mirroring pipeline scripts …"
$trocrScripts = @("finetune_trocr.py", "generate_trocr_dataset.py", "label_pdfs.py")
foreach ($s in $trocrScripts) {
    $src = Join-Path $trocrDir $s
    if (-not (Test-Path -LiteralPath $src)) {
        Write-Warning "Missing pipeline script (will skip): $src"
        continue
    }
    Copy-Item -LiteralPath $src -Destination $SpaceRepoRoot -Force
}

# ---------------------------------------------------------------------------
# 4. Copy fonts (always required) + pdfs (optional, can be empty)
# ---------------------------------------------------------------------------
$spaceFonts = Join-Path $SpaceRepoRoot "fonts"
$spacePdfs  = Join-Path $SpaceRepoRoot "pdfs"
New-Item -ItemType Directory -Force -Path $spaceFonts | Out-Null
New-Item -ItemType Directory -Force -Path $spacePdfs  | Out-Null

Write-Host "[4/5] Copying fonts …"
Get-ChildItem -LiteralPath $fontsDir -File -Include "*.ttf","*.otf" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $spaceFonts -Force
}
$fontCount = (Get-ChildItem -LiteralPath $spaceFonts -File).Count
Write-Host "      fonts copied: $fontCount"
if ($fontCount -eq 0) { throw "No .ttf/.otf fonts in $fontsDir - dataset gen would fail." }

if (Test-Path -LiteralPath $pdfsDir) {
    Get-ChildItem -LiteralPath $pdfsDir -File -Include "*.pdf","*.png","*.jpg","*.jpeg" -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $spacePdfs -Force }
    $pdfCount = (Get-ChildItem -LiteralPath $spacePdfs -File).Count
    Write-Host "      pdfs/images copied: $pdfCount (optional, may be 0)"
} else {
    Write-Host "      training/pdfs/ not present - Space pdfs/ will be empty (synthetic-only run)."
}

# ---------------------------------------------------------------------------
# 5. Optional: git init + remote
# ---------------------------------------------------------------------------
if ($InitGit) {
    Write-Host "[5/5] Initializing git in $SpaceRepoRoot …"
    & git -C $SpaceRepoRoot init -b main
    if ($LASTEXITCODE -ne 0) { throw "git init failed (is git installed?)" }

    # git-lfs is required by HF Hub: it rejects any individual file >10 MiB
    # pushed as a regular blob, and also rejects binary file types it expects
    # to see via LFS (fonts, pdfs, images, .safetensors, etc.). Track every
    # binary the Space repo will contain so the first push doesn't get bumped
    # by the pre-receive hook.
    $lfsCheck = & git lfs version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git-lfs not installed (required by HF Hub). Install from https://git-lfs.github.com/"
    }
    & git -C $SpaceRepoRoot lfs install
    if ($LASTEXITCODE -ne 0) { throw "git lfs install failed" }

    # .gitattributes lives at the repo root and lists LFS-tracked patterns.
    # We write it before `git add` so the smudge/clean filters engage on the
    # initial add - that's what turns the binaries into LFS pointers in the
    # commit instead of bloating the repo.
    $gitAttrs = Join-Path $SpaceRepoRoot ".gitattributes"
    $lfsPatterns = @(
        "fonts/*.ttf",
        "fonts/*.otf",
        "pdfs/*.pdf",
        "pdfs/*.png",
        "pdfs/*.jpg",
        "pdfs/*.jpeg"
    )
    foreach ($p in $lfsPatterns) {
        & git -C $SpaceRepoRoot lfs track $p | Out-Null
    }
    # Make sure .gitattributes itself is staged before the rest of the files
    # so the LFS filters apply to everything added afterwards.
    & git -C $SpaceRepoRoot add .gitattributes
    if ($LASTEXITCODE -ne 0) { throw "git add .gitattributes failed" }

    if ($HfUsername) {
        $remote = "https://huggingface.co/spaces/$HfUsername/$SpaceName"
        $existing = & git -C $SpaceRepoRoot remote 2>&1
        if ($existing -contains "origin") {
            & git -C $SpaceRepoRoot remote remove origin
        }
        & git -C $SpaceRepoRoot remote add origin $remote
        if ($LASTEXITCODE -eq 0) {
            Write-Host "      origin -> $remote"
        } else {
            throw "git remote add failed"
        }
    } else {
        Write-Host "      (skip remote - pass -HfUsername <name> to wire origin)"
    }
} else {
    Write-Host "[5/5] (skip git init - re-run with -InitGit to enable)"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Done. Tree:"
Get-ChildItem -LiteralPath $SpaceRepoRoot -Recurse -File |
    ForEach-Object { "  " + $_.FullName.Substring($SpaceRepoRoot.Length).TrimStart('\') }
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Create a HF Space (SDK: docker, hardware: T4 small)."
Write-Host "     Use the same name you passed to -SpaceName ($SpaceName) so the"
Write-Host "     pre-wired remote matches."
Write-Host "  2. In Space → Settings → Repository secrets, set:"
Write-Host "       HF_TOKEN      (write-scoped token)"
Write-Host "       HF_USERNAME   ($HfUsername)"
Write-Host "     And optionally a Repository variable (or secret):"
Write-Host "       HUB_MODEL_ID  = $HfUsername/$SpaceName"
Write-Host "     (or leave blank and type `$`{HF_USERNAME}/$SpaceName in the UI text box"
Write-Host "     - the app resolves the placeholder against HF_USERNAME)."
Write-Host "  3. From the space repo:"
Write-Host "       cd `"$SpaceRepoRoot`""
if ($InitGit) {
    Write-Host "       git add -A"
    Write-Host "       git commit -m `"Initial TrOCR Space ($SpaceName)`""
    Write-Host "       git push -u origin main"
} else {
    Write-Host "       git init -b main"
    Write-Host "       git remote add origin https://huggingface.co/spaces/$HfUsername/$SpaceName"
    Write-Host "       git add -A ; git commit -m `"Initial TrOCR Space ($SpaceName)`""
    Write-Host "       git push -u origin main"
}
Write-Host "  4. Open the Space URL. Click 1. Generate, then 3. Run fine-tuning."
Write-Host "     Model lands at: $HfUsername/$SpaceName"