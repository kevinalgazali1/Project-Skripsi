<?php

namespace App\Http\Controllers;

use App\Models\Artikel;
use Illuminate\Support\Str;
use Illuminate\Http\Request;

class ArtikelController extends Controller
{
    public function store(Request $request)
    {
        try {
            $request->validate([
                'judul' => 'required|string|max:255|unique:artikels,judul',
                'isi' => 'required',
                'penulis' => 'required|string|max:255',
                'gambar_artikel' => 'nullable|image|mimes:jpeg,png,jpg|max:10048',
            ]);

            $user_id = $request->user()->id;

            // Proses file gambar jika ada
            $file_path = null;
            if ($request->hasFile('gambar_artikel')) {
                $file = $request->file('gambar_artikel');
                $file_name = $file->hashName(); // nama unik
                $file->move(public_path('storage/assets/artikel'), $file_name);
                $file_path = 'assets/artikel/' . $file_name;
            }

            // Simpan data ke database
            Artikel::create([
                'user_id' => $user_id,
                'judul' => $request->judul,
                'slug' => Str::slug($request->judul),
                'isi' => $request->isi,
                'penulis' => $request->penulis,
                'gambar_artikel' => $file_path,
            ]);

            return redirect()->back()->with('success', 'Artikel berhasil ditambahkan.');
        } catch (\Exception $e) {
            return redirect()->back()->with('danger', 'Artikel gagal ditambahkan.');
        }
    }


    public function update(Request $request, $id)
    {
        $artikel = Artikel::findOrFail($id);

        $request->validate([
            'judul' => 'required|string|max:255',
            'penulis' => 'required|string|max:255',
            'isi' => 'required',
            'gambar_artikel' => 'nullable|image|max:10048',
        ]);

        $artikel->judul = $request->judul;
        $artikel->penulis = $request->penulis;
        $artikel->isi = $request->isi;

        // 👇 Bagian update gambar
        if ($request->hasFile('gambar_artikel')) {
            $oldPath = public_path('storage/' . $artikel->gambar_artikel);
            if (file_exists($oldPath)) {
                unlink($oldPath);
            }

            $file = $request->file('gambar_artikel');
            $file_name = $file->hashName();
            $file->move(public_path('storage/assets/artikel'), $file_name);

            $artikel->gambar_artikel = 'assets/artikel/' . $file_name;
        }

        $artikel->save();

        return redirect()->back()->with('success', 'Artikel berhasil diperbarui.');
    }

    public function destroy($id)
    {
        try {
            $artikel = Artikel::findOrFail($id);

            // Hapus gambar jika ada
            $gambarPath = public_path('storage/' . $artikel->gambar_artikel);
            if (file_exists($gambarPath)) {
                unlink($gambarPath);
            }

            $artikel->delete();

            return redirect()->back()->with('success', 'Artikel berhasil dihapus.');
        } catch (\Exception $e) {
            return redirect()->back()->with('error', 'Gagal menghapus artikel: ' . $e->getMessage());
        }
    }
}
