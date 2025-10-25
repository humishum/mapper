# 3D Pointcloud Visualizer - Documentation Index

## 🎯 Start Here

**New to this project?** → [QUICKSTART.md](QUICKSTART.md) (5 min read)

**Want to understand everything?** → [README.md](README.md) (15 min read)

## 📚 Documentation Guide

### For Users

| Document | Purpose | Time | When to Read |
|----------|---------|------|--------------|
| [QUICKSTART.md](QUICKSTART.md) | Get up and running fast | 5 min | First time setup |
| [README.md](README.md) | Complete user guide | 15 min | Want full details |
| [SUMMARY.md](SUMMARY.md) | High-level overview | 5 min | Quick reference |

### For Developers

| Document | Purpose | Time | When to Read |
|----------|---------|------|--------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and internals | 20 min | Before modifying code |
| [MIGRATION.md](MIGRATION.md) | Changes from old version | 10 min | If you used old version |

## 🚀 Quick Commands

```bash
# First time setup (Node.js)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 18

# Install dependencies
cd visualizer/frontend && npm install

# Export data
cd visualizer
./scripts/export_data.sh

# Start visualizer
./scripts/dev.sh
```

## 📁 Code Organization

```
visualizer/
├── backend/           → Python data processing
├── frontend/          → JavaScript visualization
├── scripts/           → Helper scripts
└── *.md              → Documentation (you are here!)
```

## 🔍 Find What You Need

### "How do I..."

- **Get started?** → [QUICKSTART.md](QUICKSTART.md)
- **Change the threshold?** → [README.md](README.md#configuration)
- **Add a new feature?** → [ARCHITECTURE.md](ARCHITECTURE.md#development-workflow)
- **Deploy to production?** → [README.md](README.md#build-for-production)
- **Understand the code?** → [ARCHITECTURE.md](ARCHITECTURE.md)

### "What is..."

- **The overall architecture?** → [ARCHITECTURE.md](ARCHITECTURE.md#system-architecture)
- **The data flow?** → [ARCHITECTURE.md](ARCHITECTURE.md#data-flow)
- **Different from the old version?** → [MIGRATION.md](MIGRATION.md)
- **The technology stack?** → [ARCHITECTURE.md](ARCHITECTURE.md#technology-stack)

### "Why did you..."

- **Use deck.gl?** → [ARCHITECTURE.md](ARCHITECTURE.md#why-deckgl)
- **Separate Python and JavaScript?** → [ARCHITECTURE.md](ARCHITECTURE.md#design-decisions)
- **Export to JSON instead of direct loading?** → [ARCHITECTURE.md](ARCHITECTURE.md#why-json-instead-of-direct-ply-loading)

## 📊 Project Stats

- **Total Documentation**: 1,400+ lines across 6 files
- **Backend Code**: ~620 lines (4 modules)
- **Frontend Code**: ~400 lines (4 modules + HTML)
- **Helper Scripts**: ~80 lines (2 scripts)
- **Total Project**: ~2,500 lines

## 🎓 Learning Path

### Beginner
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Follow the setup steps
3. Explore the UI
4. Try different control settings

### Intermediate
1. Read [README.md](README.md)
2. Understand the command-line options
3. Export data with different parameters
4. Look at the generated data.json

### Advanced
1. Read [ARCHITECTURE.md](ARCHITECTURE.md)
2. Understand the component design
3. Explore the source code
4. Make modifications
5. Add new features

## 🛠 Maintenance

### Regular Tasks
- Update Node.js: `nvm install --lts`
- Update dependencies: `npm update` (in frontend/)
- Re-export data after source changes

### Troubleshooting
1. Check [README.md](README.md#troubleshooting)
2. Look at browser console (F12)
3. Check terminal output
4. Verify data.json exists and is valid

## 🎉 Features at a Glance

**Data Processing**
- ✅ Load metadata from folders
- ✅ Find PLY files by threshold/sequence
- ✅ Parse binary and ASCII PLY
- ✅ Automatic downsampling
- ✅ GPS to Cartesian conversion
- ✅ Export to JSON

**Visualization**
- ✅ High-performance WebGL rendering
- ✅ Interactive 3D camera controls
- ✅ Real-time point size adjustment
- ✅ Scale and opacity controls
- ✅ Multiple camera presets
- ✅ Location information panel

**Development**
- ✅ Hot module reload
- ✅ Modern build system (Vite)
- ✅ Clean modular architecture
- ✅ Comprehensive documentation
- ✅ Helper scripts for common tasks

## 📝 Quick Reference

### Export Data
```bash
python -m backend.export_data \
  --data-dir PATH \
  --threshold 2.0 \
  --max-points 50000 \
  --sequence-id 1
```

### Start Dev Server
```bash
cd frontend
npm run dev
```

### Build for Production
```bash
cd frontend
npm run build
```

## 🔗 Related Files

- `package.json` - Frontend dependencies
- `vite.config.js` - Build configuration
- `.nvmrc` - Node version specification
- `.gitignore` - Git ignore rules

## 💡 Pro Tips

1. Use `./scripts/*.sh` for common tasks
2. Keep data.json in gitignore (it's large)
3. Adjust max-points based on your hardware
4. Use camera presets for consistent views
5. Export once, iterate on frontend (hot reload)

---

**Need Help?** Start with [QUICKSTART.md](QUICKSTART.md) or [README.md](README.md)

**Want to Contribute?** Read [ARCHITECTURE.md](ARCHITECTURE.md) first

**Questions?** Check the troubleshooting sections in each guide

