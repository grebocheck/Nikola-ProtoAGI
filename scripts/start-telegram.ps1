param(
    [string]$Token = "",
    [string]$AllowedChatId = "",
    [ValidateSet("smart", "always", "mention", "silent")]
    [string]$ReplyMode = "smart",
    [switch]$NoProactive,
    [switch]$DeleteWebhook,
    [switch]$DropPendingUpdates
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Root "src"

$ProtoArgs = @("telegram", "--reply-mode", $ReplyMode)
if ($Token -ne "") {
    $ProtoArgs += @("--token", $Token)
}
if ($AllowedChatId -ne "") {
    $ProtoArgs += @("--allowed-chat-id", $AllowedChatId)
}
if ($NoProactive) {
    $ProtoArgs += "--no-proactive"
}
if ($DeleteWebhook) {
    $ProtoArgs += "--delete-webhook"
}
if ($DropPendingUpdates) {
    $ProtoArgs += "--drop-pending-updates"
}

python -m protoagi @ProtoArgs
