HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$HERE/local.env" ]; then
  while IFS='=' read -r k v; do
    [[ -z "$k" || "$k" == \#* ]] && continue
    [ -z "${!k}" ] && export "$k=$(eval echo "$v")"
  done < "$HERE/local.env"
fi
export ISAACLAB_DIR=${ISAACLAB_DIR:-$HOME/IsaacLab}
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
SCENE=${SCENE:-$HERE/warehouse.usda}
POLICY=${POLICY:-$HERE/policies/go2_flat/model_1499.pt}
NOISE="PhysX warning|swapchain == NULL"
for k in "$HERE"/apps/*.kit; do
  [ -f "$ISAACLAB_DIR/apps/$(basename "$k")" ] || cp "$k" "$ISAACLAB_DIR/apps/"
done
cd "$HERE"
