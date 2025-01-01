const fs = require('fs-extra');

(async () => {
    const sourceDir = './models';
    const destDir = './dist/models';

    try {
        // Ensure the destination directory exists
        await fs.ensureDir(destDir);

        // Copy files from source to destination
        await fs.copy(sourceDir, destDir);
        console.log('Models successfully copied to dist/models.');
    } catch (err) {
        console.error('Error copying models:', err);
    }
})();
