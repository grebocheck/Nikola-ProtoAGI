param(
    [string]$Token = "",
    [string]$AllowedChatId = "",
    [ValidateSet("smart", "always", "mention", "silent")]
    [string]$ReplyMode = "smart",
    [switch]$NoProactive,
    [switch]$Once,
    [switch]$DeleteWebhook,
    [switch]$DropPendingUpdates
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Root "src"

$ProtoArgs = @("telegram")
if ($PSBoundParameters.ContainsKey("ReplyMode")) {
    $ProtoArgs += @("--reply-mode", $ReplyMode)
}
if (-not [string]::IsNullOrWhiteSpace($Token)) {
    $ProtoArgs += @("--token", $Token)
}
if (-not [string]::IsNullOrWhiteSpace($AllowedChatId)) {
    $ProtoArgs += @("--allowed-chat-id", $AllowedChatId)
}
if ($NoProactive) {
    $ProtoArgs += "--no-proactive"
}
if ($Once) {
    $ProtoArgs += "--once"
}
if ($DeleteWebhook) {
    $ProtoArgs += "--delete-webhook"
}
if ($DropPendingUpdates) {
    $ProtoArgs += "--drop-pending-updates"
}

python -m protoagi @ProtoArgs
